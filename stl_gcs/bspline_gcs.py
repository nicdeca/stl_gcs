import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

from math import floor

import warnings

from pydrake.all import (
    ConvexSet,
    VPolytope,
    HPolyhedron,
    BsplineTrajectory_,
    BsplineBasis_,
    BsplineTrajectory,
    BsplineBasis,
    KnotVectorType,
    MakeMatrixContinuousVariable,
    Expression,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    MosekSolver,
    GurobiSolver,
    SolverOptions,
    CommonSolverOption,
    DecomposeLinearExpressions,
    LinearEqualityConstraint,
    Constraint,
    L1NormCost,
    L2NormCost,
    QuadraticCost,
    Binding,
    Cost,
)


from stl_gcs.graph import Graph


class BsplineGraphOfConvexSets(Graph):
    """
    This implementation is taken from:
            https://github.com/vincekurtz/ltl_gcs/blob/main/ltlgcs/bezier_gcs.py

    Problem setup and solver for planning a piecewise bezier curve trajectory
    through a graph of convex sets. The graph setup is as follows:

        - Each vertex is associated with a convex set
        - Each convex set contains a Bezier curve
        - The optimal path is a sequence of Bezier curves. These curves
          must satisfy continuity (and continuous differentiability) constraints.
        - The goal is to find a (minimum cost) trajectory from a given starting
          point to the target vertex
        - The target vertex is not associated with any constraints on the
          curve: it just indicates that the task is complete.
    """

    def __init__(
        self,
        vertices: list[int],
        edges: list[tuple[int, int]],
        regions: dict[int, ConvexSet],
        start_vertex: int,
        end_vertex: int,
        start_point: np.ndarray,
        order: int = 2,
        continuity: int = 1,
    ):
        """
        Construct a graph of convex sets

        Args:
            vertices:      list of integers representing each vertex in the graph
            edges:         list of pairs of integers (vertices) for each edge
            regions:       dictionary mapping each vertex to a Drake ConvexSet
            start_vertex:  index of the starting vertex
            end_vertex:    index of the end/target vertex
            start_point:   initial point of the path
            order:         order of bezier curve under consideration
            continuity:    number of continuous derivatives of the curve
        """
        # General graph constructor
        super().__init__(vertices, edges)

        # Dimensionality of the problem is defined by the starting point
        assert regions[start_vertex].PointInSet(start_point)
        self.start_point = start_point
        self.dim = len(start_point)

        # Check that the regions correspond to valid vertices and valid convex
        # sets
        for vertex, region in regions.items():
            assert vertex in self.vertices, "invalid vertex index"
            assert isinstance(region, ConvexSet), "regions must be convex sets"
            assert region.ambient_dimension() == self.dim
        self.regions = regions

        # Validate the start and target vertices
        assert start_vertex in vertices
        assert end_vertex in vertices
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex

        # Bezier curves can guarantee continuity of n-1 derivatives
        assert continuity < order
        self.order = order
        self.continuity = continuity

        # Create "dummy" symbolic curves for an arbitrary edge. This allows us
        # to derive expressions for various things, such as derivatives of the
        # spline, in terms of the original control points (decision variables)
        # The created symbolic functions then will be fed with the proper
        # numerical values online
        # MakeMatrixContinuousVariable: creates a dynamically-sized Eigen
        # matrix of symbolic continuous variables.
        self.dummy_xu = MakeMatrixContinuousVariable(self.order + 1, self.dim, "xu")
        self.dummy_xv = MakeMatrixContinuousVariable(self.order + 1, self.dim, "xv")
        self.dummy_edge_vars = np.concatenate(
            (self.dummy_xu.flatten(), self.dummy_xv.flatten())
        )

        self.dummy_path_u = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0, 1
            ),
            self.dummy_xu.T,
        )
        self.dummy_path_v = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0, 1
            ),
            self.dummy_xv.T,
        )

        # Create the GCS problem
        self.gcs = GraphOfConvexSets()
        self.source, self.target = self.SetupShortestPathProblem()

    # -------------------------------------------------------
    # -------------------- AddLengthCost --------------------

    def AddLengthCost(self, weight: float = 1.0, norm: str = "L2"):
        """
        Add to each edge a penalty on the distance between control points,
        which is an overapproximation of total path length.

        There are several norms we can use to approximate these lengths:
            L1 - most computationally efficient, and introduces only linear
            costs and constraints to the GCS problem.

            L2 - closest approximation to the actual curve length, introduces
            cone constraints to the GCS problem.

            L2_squared - incentivises evenly space control points, which can
            produce some nice smooth-looking curves. Introduces cone constraints
            to the GCS problem.

        Args:
            weight: Weight for this cost, scalar.
            norm: Norm to use to when evaluating distance between control
                  points. See AddDerivativeCost for details.
        """
        assert norm in ["L1", "L2", "L2_squared"], "invalid length norm"

        # Get a symbolic expression for the difference between subsequent
        # control points, in terms of the original decision variables
        control_points = self.dummy_path_u.control_points()

        A = []
        for i in range(self.order):
            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff = control_points[i] - control_points[i + 1]
            A.append(
                DecomposeLinearExpressions(diff.flatten(), self.dummy_xu.flatten())
            )

        # Apply a cost to the starting segment of each edge.
        for edge in self.gcs.Edges():
            x = edge.xu()
            for i in range(self.order):
                if norm == "L1":
                    cost = L1NormCost(weight * A[i], np.zeros(self.dim))
                elif norm == "L2":
                    cost = L2NormCost(weight * A[i], np.zeros(self.dim))
                else:  # L2 squared
                    cost = QuadraticCost(
                        Q=weight * A[i].T @ A[i], b=np.zeros(len(x)), c=0.0
                    )
                edge.AddCost(Binding[Cost](cost, x))

    # -----------------------------------------------------------
    # -------------------- AddDerivativeCost --------------------

    def AddDerivativeCost(self, degree: int, weight: float = 1.0, norm: str = "L2"):
        """
        Add a penalty on the derivative of the path. We do this by penalizing
        some norm of the control points of the derivative of the path. Notice
        that we are not minimizing derivatives wrt time, but rather the
        derivative of the Bezier curve wrt its parameter. This is still a
        reasonable proxy for smoothness of the path.

        Args:
            degree: The derivative to penalize. degree=0 is the original
                    trajectory, degree=1 is the first derivative. Notice that
                    we can only penalize derivatives up to order-1.
            weight: Weight for this cost, scalar
            norm:   Norm to use to when evaluating distance between control
                    points (see above)
        """
        assert norm in ["L1", "L2", "L2_squared"], "invalid length norm"
        assert degree >= 0
        assert degree < self.order

        # Get a symbolic version of the i^th derivative of a segment
        path_deriv = self.dummy_path_u.MakeDerivative(degree)

        # Get a symbolic expression for the difference between subsequent
        # control points in the i^th derivative, in terms of the original
        # decision variables
        deriv_control_points = path_deriv.control_points()

        A = []
        for i in range(self.order - degree + 1):
            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff = deriv_control_points[i]  # - deriv_control_points[i+1]
            A.append(
                DecomposeLinearExpressions(diff.flatten(), self.dummy_xu.flatten())
            )

        # Apply a cost to the starting segment of each edge.
        for edge in self.gcs.Edges():
            x = edge.xu()
            for i in range(self.order - degree + 1):
                if norm == "L1":
                    cost = L1NormCost(weight * A[i], np.zeros(self.dim))
                elif norm == "L2":
                    cost = L2NormCost(weight * A[i], np.zeros(self.dim))
                else:  # L2 squared
                    cost = QuadraticCost(
                        Q=weight * A[i].T @ A[i], b=np.zeros(len(x)), c=0.0
                    )
                edge.AddCost(Binding[Cost](cost, x))

    # ------------------------------------------------------------------
    # -------------------- SetupShortestPathProblem --------------------

    def SetupShortestPathProblem(self):
        """
        Formulate a shortest path through convex sets problem where the path is
        composed of bezier curves that must be contained in each convex set.

        Returns:
            source: Drake Gcs Vertex corresponding to the initial convex set
            target: Drake Gcs Vertex corresponding to the final convex set
        """
        # Define vertices. The convex sets for each vertex are such that each
        # control point must be contained in the corresponding region
        gcs_verts = {}  # map our vertices to GCS vertices
        for v in self.vertices:
            gcs_verts[v] = self.gcs.AddVertex(
                self.regions[v].CartesianPower(self.order + 1)
            )

        # Define edges
        for e in self.edges:
            # Get vertex IDs of source and target for this edge
            u = gcs_verts[e[0]]
            v = gcs_verts[e[1]]

            self.gcs.AddEdge(u, v)

        # Define source and target vertices
        source = gcs_verts[self.start_vertex]
        target = gcs_verts[self.end_vertex]

        # Add continuity constraints. This includes both continuity (first and
        # last control points line up) and smoothness (first and last control
        # points of derivatives of the path line up).
        for i in range(self.continuity + 1):
            # N.B. i=0 corresponds to the path itsefl
            dummy_path_u_deriv = self.dummy_path_u.MakeDerivative(i)
            dummy_path_v_deriv = self.dummy_path_v.MakeDerivative(i)

            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter("ignore", category=RuntimeWarning)
                continuity_err = (
                    dummy_path_v_deriv.control_points()[0]
                    - dummy_path_u_deriv.control_points()[-1]
                )

            continuity_constraint = LinearEqualityConstraint(
                DecomposeLinearExpressions(continuity_err, self.dummy_edge_vars),
                np.zeros(self.dim),
            )

            # Apply the continuity constraints to each edge in the graph
            for edge in self.gcs.Edges():
                if edge.v() != target:
                    edge.AddConstraint(
                        Binding[Constraint](
                            continuity_constraint,
                            np.concatenate((edge.xu(), edge.xv())),
                        )
                    )

        # Add initial condition constraint
        for i in range(self.dim):
            source.AddConstraint(source.x()[i] == self.start_point[i])

        # Allow access to GCS vertices later
        self.gcs_verts = gcs_verts

        return (source, target)

    # -----------------------------------------------------------
    # -------------------- SolveShortestPath --------------------

    def SolveShortestPath(
        self,
        verbose: bool = True,
        convex_relaxation: bool = False,
        preprocessing: bool = True,
        max_rounded_paths: int = 0,
        solver: str = "mosek",
    ):
        """
        Solve the shortest path problem (self.gcs).

        Args:
            verbose: whether to print solver details to the screen
            convex_relaxation: whether to solve the original MICP or the convex
                               relaxation (+rounding)
            preprocessing: preprocessing step to reduce the size of the graph
            max_rounded_paths: number of distinct paths to compare during
                               rounding for the convex relaxation
            solver: underling solver for the CP/MICP. Must be "mosek" or
                    "gurobi"

        Returns:
            result: a MathematicalProgramResult encoding the solution.
        """
        # Set solver options
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = convex_relaxation
        options.preprocessing = preprocessing
        options.max_rounded_paths = max_rounded_paths
        if solver == "mosek":
            options.solver = MosekSolver()
        elif solver == "gurobi":
            options.solver = GurobiSolver()
        else:
            raise ValueError(f"Unknown solver {solver}")
        solver_opts = SolverOptions()
        solver_opts.SetOption(CommonSolverOption.kPrintToConsole, verbose)
        options.solver_options = solver_opts

        # Solve the problem
        result = self.gcs.SolveShortestPath(self.source, self.target, options)
        if not result.is_success():
            print("GCS failed to find a solution")
            return result

        # If we solved the convex relaxation, refine the solution using
        # convex restriction
        # if convex_relaxation:
        #     # Get the optimal path
        #     path = self.gcs.GetSolutionPath(self.source, self.target, result)
        #     result = self.gcs.SolveConvexRestriction(
        #         path, options=options, initial_guess=result
        #     )

        return result

    # -----------------------------------------------------------
    # -------------------- PlotScenario -------------------------

    def PlotScenario(self):
        """
        Add a plot of each region to the current matplotlib axes.
        Only supports 2D polytopes for now.
        """
        for vertex, region in self.regions.items():
            assert region.ambient_dimension() == 2, "only 2D sets allowed"

            if vertex != self.end_vertex:
                # The target vertex is trivial and therefore not plotted

                # Compute vertices of the polygon in known order
                v = VPolytope(region).vertices().T
                hull = ConvexHull(v)
                v_sorted = np.vstack([v[hull.vertices, 0], v[hull.vertices, 1]]).T

                # Make a polygonal patch
                poly = Polygon(v_sorted, alpha=0.5, edgecolor="k", linewidth=3)
                plt.gca().add_patch(poly)

        # Use equal axes so square things look square
        plt.axis("equal")

    # -----------------------------------------------------------
    # -------------------- ExtractSolution ----------------------

    def ExtractSolution(self, result):
        """
        Extract a sequence of bezier curves representing the optimal solution.

        Args:
            result: MathematicalProgramResult from calling SolveShortestPath

        Returns:
            A list of BsplineTrajectory objects representing each segment of
            the optimal solution.
        """
        # List of bezier curves that we'll return
        curves = []

        def get_outgoing_edge(vertex):
            # Helper function that returns the highest probability (phi-value)
            # edge leading out of the given vertex
            edge = None
            phi = -1.0
            for e in self.gcs.Edges():
                if (e.u() == vertex) and result.GetSolution(e.phi()) > phi:
                    edge = e
                    phi = result.GetSolution(e.phi())
            return edge

        # Traverse the graph along the optimal path
        v = self.gcs_verts[self.start_vertex]
        while v != self.gcs_verts[self.end_vertex]:
            e = get_outgoing_edge(v)

            # Get the control points for this vertex
            xu = result.GetSolution(e.xu())
            control_points = xu.reshape(self.order + 1, -1)
            basis = BsplineBasis(
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0, 1
            )
            curves.append(BsplineTrajectory(basis, control_points.T))

            # move to the next vertex
            v = e.v()

        return curves

    # -----------------------------------------------------------
    # -------------------- AnimateSolution ----------------------

    def AnimateSolution(self, result, show=True, save=False, filename=None):
        """
        Create an animation of the solution on the current set of matplotlib
        axes.

        Args:
            result:   MathematicalProgramResult from calling SolveShortestPath
            show:     Flag for displaying the animation immediately
            save:     Flag for saving a gif of the animation
            filename: String denoting what file to save the animation to (.gif)
        """
        assert self.dim == 2, "animation only supported in 2D"

        # Get current matplotlib figure and axes
        fig = plt.gcf()
        ax = plt.gca()

        # Get the solution as a sequence of splines
        s = self.ExtractSolution(result)
        q = ax.scatter(*s[0].value(0), color="blue", s=50, zorder=3)

        def animate(t):
            segment = floor(t)
            new_q = s[segment].value(t % 1).T
            q.set_offsets(new_q)
            return q

        t = np.arange(0, len(s), 0.02)
        ani = animation.FuncAnimation(fig, animate, t, interval=50, blit=False)

        if save:
            assert filename is not None, "must supply a filename to save the animation"
            print(f"Saving animation to {filename}, this may take a minute...")
            ani.save(filename, writer=animation.PillowWriter(fps=30))

        if show:
            plt.show()

        return ani

    # -----------------------------------------------------------
    # -------------------- PlotSolution -------------------------

    def PlotSolution(self, result, plot_control_points=True, plot_path=True):
        """
        Add a plot of the solution to the current matplotlib axes. Only
        supported for 2D.

        Args:
            result: MathematicalProgramResult from calling SolveShortestPath
            plot_control_points: flag for plotting the control points
            plot_path: flag for plotting the actual path
        """
        for edge in self.gcs.Edges():
            # Note that focusing on xu for each edge ignores the target
            # state, since that's just an indicator of task completion
            phi = result.GetSolution(edge.phi())
            xu = result.GetSolution(edge.xu())

            if phi > 0.0:
                # Construct a bezier curve from the control points
                control_points = xu.reshape(self.order + 1, -1)
                basis = BsplineBasis(
                    self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0, 1
                )
                path = BsplineTrajectory(basis, control_points.T)

                if plot_control_points:
                    plt.plot(
                        control_points[:, 0],
                        control_points[:, 1],
                        "o--",
                        color="red",
                        alpha=phi,
                    )

                if plot_path:
                    curve = path.vector_values(np.linspace(0, 1))
                    plt.plot(
                        curve[0, :], curve[1, :], color="blue", linewidth=3, alpha=phi
                    )


# ============================================================
# -------------------- Example usage -------------------------
# ============================================================

if __name__ == "__main__":

    def toy_example():

        # Utility: plot a VPolytope
        def plot_vpoly(ax, vpoly, name=None, color="lightblue"):
            verts = vpoly.vertices().T

            hull = ConvexHull(verts)
            v_sorted = np.vstack([verts[hull.vertices, 0], verts[hull.vertices, 1]]).T

            # Make a polygonal patch
            poly = Polygon(v_sorted, alpha=0.5, edgecolor="k", linewidth=3)
            plt.gca().add_patch(poly)

        # ---- Step 1. Define polytopic sets (boxes in 2D) ----
        # 1) Build polytopic "rooms" + corridor
        # Notice that the class polytope does not have the CartesianPower method so we have to use HPolyhedron
        # Notice that one can get a Vpolytope from an HPolyhedron using the constructor VPolytope(HPolyhedron)
        # and viceversa (admitting that the Hpolyhedron is bounded, i.e. a polytope)
        Sid = 0
        S = HPolyhedron.MakeBox([0.0, 0.0], [1.0, 1.0])  # [0,1] × [0,1]

        R1id = 1
        R1 = HPolyhedron.MakeBox([1.0, 0.0], [2.0, 1.0])  # attached to S at x=1

        R2id = 2
        R2 = HPolyhedron.MakeBox([2.0, 0.0], [3.0, 1.0])  # attached to R1 at x=2

        Cid = 3
        C = HPolyhedron.MakeBox([3.0, 0.0], [4.0, 1.0])  # attached to R2 at x=3

        R3id = 4
        R3 = HPolyhedron.MakeBox([4.0, 0.0], [5.0, 1.0])  # attached to C at x=4

        Tid = 5
        T = HPolyhedron.MakeBox([5.0, 0.0], [6.0, 1.0])  # attached to R3 at x=5

        # --- Plot everything ---
        fig, ax = plt.subplots()
        plot_vpoly(ax, VPolytope(S), "S", "lightgreen")
        plot_vpoly(ax, VPolytope(R1), "R1", "lightblue")
        plot_vpoly(ax, VPolytope(R2), "R2", "orange")
        plot_vpoly(ax, VPolytope(C), "C", "lightgrey")
        plot_vpoly(ax, VPolytope(R3), "R3", "violet")
        plot_vpoly(ax, VPolytope(T), "T", "pink")

        ax.set_aspect("equal")
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 2.0)
        ax.axis("off")
        plt.show()

        # Define the graph structure
        vertices = [Sid, R1id, R2id, Cid, R3id, Tid]
        edges = [
            (Sid, R1id),
            (R1id, R2id),
            (R2id, Cid),
            (Cid, R3id),
            (R3id, Tid),
            (R1id, Cid),
            (R2id, R3id),
        ]

        # Define the convex sets associated with each vertex
        regions = {
            Sid: S,
            R1id: R1,
            R2id: R2,
            Cid: C,
            R3id: R3,
            Tid: T,
        }

        start_vertex = Sid
        end_vertex = Tid
        start_point = np.array([0.5, 0.5])

        # Create the Bspline GCS problem
        gcs = BsplineGraphOfConvexSets(
            vertices,
            edges,
            regions,
            start_vertex,
            end_vertex,
            start_point,
            order=3,
            continuity=2,
        )

        # Add costs to the problem
        gcs.AddLengthCost(weight=1.0, norm="L2_squared")
        gcs.AddDerivativeCost(degree=1, weight=10.0, norm="L2_squared")

        # Plot the scenario
        plt.figure(figsize=(8, 8))
        gcs.PlotScenario()
        plt.title("Scenario")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-1, 7)
        plt.ylim(-1, 7)
        plt.grid()
        plt.pause(0.1)  # pause to ensure the plot updates

        # Solve the problem
        result = gcs.SolveShortestPath(
            verbose=True,
            convex_relaxation=True,
            preprocessing=True,
            max_rounded_paths=5,
            solver="mosek",
        )
        assert result.is_success()
        print(f"Optimal cost: {result.get_optimal_cost()}")

        # Plot the solution
        gcs.PlotSolution(result, plot_control_points=True, plot_path=True)
        plt.title("Optimal solution")
        plt.pause(0.1)  # pause to ensure the plot updates

        # Animate the solution
        gcs.AnimateSolution(result, show=True, save=False, filename=None)

    toy_example()
