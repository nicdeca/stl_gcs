import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
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
    MakeVectorContinuousVariable,
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
    LinearCost,
    L1NormCost,
    L2NormCost,
    QuadraticCost,
    Binding,
    Cost,
    PerspectiveQuadraticCost,
    LinearConstraint,
)


from stl_gcs.graph import Graph


class SpatioTemporalBsplineGraphOfConvexSets(Graph):
    """
    Problem setup and solver for planning a piecewise bezier curve trajectory
    through a graph of convex sets. The graph setup is as follows:

        - Each vertex is associated with a convex set
        - Each convex set contains a Bezier curve
        - The optimal path is a sequence of Bezier curves. These curves
          must satisfy continuity (and continuous differentiability) constraints.
        - The goal is to find a (minimum cost) trajectory from a given starting
          point to the target vertex
        - The target vertex is just a dummy vertex not associated with any
          constraints on the curve: it just indicates that the task is complete.
        - Instead, differently from other GCS formulations, the starting
          vertex is associated with a set and the related constraints

    Here, we have a bspline associated with space and a bspline associated with time.
    "r" refers to the spatial part of the trajectory, "p" refers to the time/duration part of the trajectory.
    """

    def __init__(
        self,
        vertices: list[int],
        edges: list[tuple[int, int]],
        regions: dict[int, ConvexSet],
        start_vertex: int,
        end_vertex: int,
        start_point: np.ndarray,
        order: int = 3,
        continuity: int = 1,
        pdot_min: float = 1e-3,
    ):
        """
        Construct a graph of convex sets

        Args:
            vertices:      list of integers representing each vertex in the graph
            edges:         list of pairs of integers (vertices) for each edge
            regions:       dictionary mapping each vertex to a Drake ConvexSet
            start_vertex:  index of the starting vertex
            end_vertex:    index of the end/target vertex
            start_point:   initial point of the path (include space and time [x,t])
            order:         order of bezier curve under consideration
            continuity:    number of continuous derivatives of the curve
            pdot_min:      minimum derivative of time wrt the spline variable.
                           Used to regularize higher order derivatives.
        """
        # General graph constructor
        super().__init__(vertices, edges)

        # Dimensionality of the problem is defined by the starting point
        assert regions[start_vertex].PointInSet(start_point)
        self.start_point = start_point
        self.spatiotemporal_dim = len(start_point)  # include time dimension
        self.dim = self.spatiotemporal_dim - 1  # last dimension is time

        # Check that the regions correspond to valid vertices and valid convex
        # sets
        for vertex, region in regions.items():
            assert vertex in self.vertices, "invalid vertex index"
            assert isinstance(region, ConvexSet), "regions must be convex sets"
            assert region.ambient_dimension() == self.spatiotemporal_dim
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

        self.pdot_min = pdot_min

        # Create decision variables + trajectories
        self._define_variables_and_trajectories()

        # Create the GCS problem
        self.gcs = GraphOfConvexSets()
        self.SetupShortestPathProblem()

    # -------------------------------------------------------
    # ------- _define_symbolic_variables_and_trajectories ---
    def _define_variables_and_trajectories(self):
        """Define symbolic control points, in space and time,
        and corresponding B-spline trajectories."""

        # Create symbolic variables for the spatial and temporal parts of the trajectory
        self.u_vars = MakeMatrixContinuousVariable(
            self.order + 1, self.spatiotemporal_dim, "xu"
        )
        self.v_vars = MakeMatrixContinuousVariable(
            self.order + 1, self.spatiotemporal_dim, "xv"
        )

        # Extract spatial and temporal parts of the variables
        self.u_spatial = self.u_vars[:, :-1]
        self.u_temporal = self.u_vars[:, -1]
        self.v_spatial = self.v_vars[:, :-1]
        self.v_temporal = self.v_vars[:, -1]

        # Edge variables (for continuity constraints)
        self.edge_vars = np.concatenate((self.u_vars.flatten(), self.v_vars.flatten()))

        # Create B-spline trajectories for the spatial and temporal parts
        self.u_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0.0, 1.0
            ),
            self.u_spatial.T,
        )
        self.u_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0.0, 1.0
            ),
            np.expand_dims(self.u_temporal, 0),
        )

        self.v_r_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0.0, 1.0
            ),
            self.v_spatial.T,
        )
        self.v_h_trajectory = BsplineTrajectory_[Expression](
            BsplineBasis_[Expression](
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0.0, 1.0
            ),
            np.expand_dims(self.v_temporal, 0),
        )

    # -----------------------------------------------------------------
    # -------------------- set_source_target_edges --------------------
    def set_source_target_edges(self):
        """Set the edges starting from the source and the ones ending
        at the target vertex."""

        self.source_edges = []
        self.target_edges = []

        # Add edges from source to all neighbors of start vertex
        for edge in self.gcs.Edges():
            if edge.u() == self.source:
                self.source_edges.append(edge)

        # Add edges from all neighbors of end vertex to target
        for edge in self.gcs.Edges():
            if edge.v() == self.target:
                self.target_edges.append(edge)

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
        control_points = self.u_r_trajectory.control_points()

        A = []
        for i in range(self.order):
            with warnings.catch_warnings():
                # ignore numpy warnings about subtracting symbolics
                warnings.simplefilter("ignore", category=RuntimeWarning)
                diff = control_points[i] - control_points[i + 1]
            # A.append(DecomposeLinearExpressions(diff.flatten(), self.u_vars.flatten()))
            A.append(DecomposeLinearExpressions(diff.flatten(), self.u_vars.flatten()))

        # Apply a cost to the starting segment of each edge.
        for edge in self.gcs.Edges():
            # if edge.u().id() == self.start_vertex:
            #     continue
            # if edge.v().id() == self.end_vertex:
            #     continue
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

    # ----------------------------------------------------------
    # -------------------- Add time cost -----------------------
    def addTimeCost(self, weight: float = 1.0):
        assert isinstance(weight, float) or isinstance(weight, int)

        u_time_control = self.u_h_trajectory.control_points()
        segment_time = u_time_control[-1] - u_time_control[0]
        time_cost = LinearCost(
            weight * DecomposeLinearExpressions(segment_time, self.u_vars.flatten())[0],
            0.0,
        )

        for edge in self.gcs.Edges():
            # if edge.u().id() == self.start_vertex:
            #     continue
            # if edge.v().id() == self.end_vertex:
            #     continue

            edge.AddCost(Binding[Cost](time_cost, edge.xu()))

    # ----------------------------------------------------------
    # -------------------- addPathEnergyCost -------------------
    def addPathEnergyCost(self, weight: float | np.ndarray):
        raise NotImplementedError("This function is currently disabled.")

        if isinstance(weight, float) or isinstance(weight, int):
            weight_matrix = weight * np.eye(self.dim)
        else:
            assert len(weight) == self.dim
            weight_matrix = np.diag(weight)

        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(
                u_path_control[ii], self.u_vars.flatten()
            )
            # h(s) = b_ctrl * self.u_vars.flatten()
            b_ctrl = DecomposeLinearExpressions(
                u_time_control[ii], self.u_vars.flatten()
            )
            # We are defining a Quadratic over linear cost, which is of the form
            # (||Ax+b||^2) / t with t>0
            H = np.vstack(
                ((self.order) * b_ctrl, np.matmul(np.sqrt(weight_matrix), A_ctrl))
            )
            eps = 1e-5
            H_reg = np.vstack([H, np.sqrt(eps) * np.eye(H.shape[1])])
            # PerspectiveQuadraticCost(H, zeros) creates a convex cost whose value (for a decision vector
            # x) is:
            # cost(x) = ∥(sqrt(W) Actrl) x∥^2 / (bctrl x)
            energy_cost = PerspectiveQuadraticCost(H_reg, np.zeros(H_reg.shape[0]))

            for edge in self.gcs.Edges():
                # if edge.u().id() == self.start_vertex:
                #     continue
                # if edge.v().id() == self.end_vertex:
                #     continue
                edge.AddCost(Binding[Cost](energy_cost, edge.xu()))

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
        path_deriv = self.u_r_trajectory.MakeDerivative(degree)

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
            A.append(DecomposeLinearExpressions(diff.flatten(), self.u_vars.flatten()))

        # Apply a cost to the starting segment of each edge.
        for edge in self.gcs.Edges():
            # if edge.u().id() == self.start_vertex:
            #     continue
            # if edge.v().id() == self.end_vertex:
            #     continue
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

    # ---------------------------------------------------------------------
    # -------------------- addDerivativeRegularization --------------------

    def addDerivativeRegularization(
        self, weight_r: float = 1.0, weight_h: float = 1.0, order: int = 1
    ):

        assert isinstance(order, int) and 2 <= order <= self.order
        weights = [weight_r, weight_h]
        for weight in weights:
            assert isinstance(weight, float) or isinstance(weight, int)

        trajectories = [self.u_r_trajectory, self.u_h_trajectory]
        for traj, weight in zip(trajectories, weights):
            derivative_control = traj.MakeDerivative(order).control_points()
            for c in derivative_control:
                A_ctrl = DecomposeLinearExpressions(c, self.u_vars.flatten())
                H = A_ctrl.T.dot(A_ctrl) * 2 * weight / (1 + self.order - order)
                reg_cost = QuadraticCost(H, np.zeros(H.shape[0]), 0)

                for edge in self.gcs.Edges():
                    # if edge.u().id() == self.start_vertex:
                    #     continue
                    # if edge.v().id() == self.end_vertex:
                    #     continue
                    edge.AddCost(Binding[Cost](reg_cost, edge.xu()))

    # ----------------------------------------------------------------------
    # -------------------- addVelocityLimits --------------------

    def addVelocityLimits(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        assert len(lower_bound) == self.dim
        assert len(upper_bound) == self.dim

        u_path_control = self.u_r_trajectory.MakeDerivative(1).control_points()
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        lb = np.expand_dims(lower_bound, 1)
        ub = np.expand_dims(upper_bound, 1)

        for ii in range(len(u_path_control)):
            A_ctrl = DecomposeLinearExpressions(
                u_path_control[ii], self.u_vars.flatten()
            )
            b_ctrl = DecomposeLinearExpressions(
                u_time_control[ii], self.u_vars.flatten()
            )
            A_constraint = np.vstack((A_ctrl - ub * b_ctrl, -A_ctrl + lb * b_ctrl))
            velocity_con = LinearConstraint(
                A_constraint,
                -np.inf * np.ones(2 * self.dim),
                np.zeros(2 * self.dim),
            )
            # self.deriv_constraints.append(velocity_con)

            for edge in self.gcs.Edges():
                # if edge.u().id() == self.start_vertex:
                #     continue
                # if edge.v().id() == self.end_vertex:
                #     continue
                edge.AddConstraint(Binding[Constraint](velocity_con, edge.xu()))

    # ----------------------------------------------------------------------
    # -------------------- addTimeMonotonicityConstraint --------------------

    def addTimeMonotonicityConstraint(self):
        """Add constraint on the control points of the derivative of the
        time spline to ensure that time is always increasing."""
        u_time_control = self.u_h_trajectory.MakeDerivative(1).control_points()
        for ii in range(len(u_time_control)):
            b_ctrl = DecomposeLinearExpressions(
                u_time_control[ii], self.u_vars.flatten()
            )
            time_con = LinearConstraint(
                b_ctrl, np.array([self.pdot_min]), np.array([np.inf])
            )

            for edge in self.gcs.Edges():
                # if edge.u().id() == self.start_vertex:
                #     continue
                edge.AddConstraint(Binding[Constraint](time_con, edge.xu()))

    # ----------------------------------------------------------------------
    # -------------------- Setup continuity constraints --------------------
    def _set_continuity_constraints(self):
        """Build constraints for continuity of trajectory and time variables."""
        continuity_constraints = []

        for deriv in range(self.continuity + 1):
            # Spatial continuity
            u_path_deriv = self.u_r_trajectory.MakeDerivative(deriv)
            v_path_deriv = self.v_r_trajectory.MakeDerivative(deriv)
            path_err = (
                v_path_deriv.control_points()[0] - u_path_deriv.control_points()[-1]
            )
            continuity_constraints.append(
                LinearEqualityConstraint(
                    DecomposeLinearExpressions(path_err, self.edge_vars),
                    np.zeros(self.dim),
                )
            )

            # Time continuity
            u_time_deriv = self.u_h_trajectory.MakeDerivative(deriv)
            v_time_deriv = self.v_h_trajectory.MakeDerivative(deriv)
            time_err = (
                v_time_deriv.control_points()[0] - u_time_deriv.control_points()[-1]
            )
            continuity_constraints.append(
                LinearEqualityConstraint(
                    DecomposeLinearExpressions(time_err, self.edge_vars), 0.0
                )
            )

        # Apply the continuity constraints to each edge in the graph
        for edge in self.gcs.Edges():
            if edge.v() != self.target:
                edge_vars = np.concatenate((edge.xu(), edge.xv()))
                for c_con in continuity_constraints:
                    edge.AddConstraint(
                        Binding[Constraint](
                            c_con,
                            edge_vars,
                        )
                    )
        return

    # ------------------------------------------------------------------
    # -------------------- addZeroDerivativeConstraints --------------------

    def addZeroDerivativeConstraints(self):
        """
        Enforce zero derivatives up to order (self.order - 1) at the start and end.
        This is done by making the first k+1 and last k+1 control points equal for
        each derivative order k.
        """

        # --- Start constraints (source vertex) ---
        for k in range(1, self.order):  # derivative order (1 .. order-1)
            for i in range(self.dim):
                # All control points [0..k] in this coordinate must be equal
                base_idx = i
                for j in range(1, k + 1):
                    self.source.AddConstraint(
                        self.source.x()[base_idx]
                        == self.source.x()[j * self.spatiotemporal_dim + i]
                    )

        # --- End constraints (target edges) ---
        for edge in self.target_edges:
            xu = edge.xu()
            for k in range(1, self.order):  # derivative order
                for i in range(self.dim):
                    # Last (k+1) control points must be equal
                    last_idx = self.spatiotemporal_dim * self.order + i
                    for j in range(1, k + 1):
                        prev_idx = self.spatiotemporal_dim * (self.order - j) + i
                        edge.AddConstraint(xu[last_idx] == xu[prev_idx])

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
        # control point must be contained in the corresponding region. This is
        # done by extending the convex set to the Cartesian power of order+1
        # copies of itself.
        gcs_verts = {}  # map our vertices to GCS vertices
        for v in self.vertices:
            if v == self.end_vertex:
                gcs_verts[v] = self.gcs.AddVertex(self.regions[v])
            else:
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
        self.source = gcs_verts[self.start_vertex]
        self.target = gcs_verts[self.end_vertex]

        self.set_source_target_edges()

        self._set_continuity_constraints()

        self.addZeroDerivativeConstraints()

        self.addTimeMonotonicityConstraint()

        # Add initial condition constraint
        for i in range(self.spatiotemporal_dim):
            # source.x() is a vector self.spatiotemporal_dim*(order+1).
            # This sets only the first control point
            self.source.AddConstraint(self.source.x()[i] == self.start_point[i])

        # Allow access to GCS vertices later
        self.gcs_verts = gcs_verts

        return

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

    def PlotScenario(
        self,
        cmap_name="tab20",
        save_fig=False,
        format="png",
        folder="results",
    ):
        """
        Plot each region (2D or 3D polytope) to the current matplotlib axes.
        """
        fig = plt.gcf()
        if self.spatiotemporal_dim == 3:
            # create 3D axis if not already present
            if not hasattr(fig, "_ax3d"):
                ax = fig.add_subplot(111, projection="3d")
                fig._ax3d = ax  # store so we reuse the same axis
            else:
                ax = fig._ax3d
        else:
            ax = plt.gca()

        cmap = cm.get_cmap(cmap_name)

        all_points = []  # collect all vertices for autoscaling

        for vertex, region in self.regions.items():
            if vertex == self.end_vertex:
                continue  # skip trivial target

            v = VPolytope(region).vertices().T
            all_points.append(v)  # store for autoscaling
            hull = ConvexHull(v)

            if self.spatiotemporal_dim == 2:
                v_sorted = np.vstack([v[hull.vertices, 0], v[hull.vertices, 1]]).T
                poly = Polygon(v_sorted, alpha=0.5, edgecolor="k", linewidth=2)
                ax.add_patch(poly)
            elif self.spatiotemporal_dim == 3:
                for simplex in hull.simplices:
                    facecolor = cmap((vertex * 37) % cmap.N)
                    tri = Poly3DCollection([v[simplex]], alpha=0.3)
                    tri.set_facecolor(facecolor)
                    tri.set_edgecolor("k")
                    ax.add_collection3d(tri)
            else:
                raise ValueError("only 2D and 3D sets allowed")

        # --- AUTOSCALE AXES ---

        all_points = np.vstack(all_points)  # shape (N, d)
        if self.spatiotemporal_dim == 2:
            ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
            ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
        elif self.spatiotemporal_dim == 3:
            ax.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
            ax.set_ylim(all_points[:, 1].min(), all_points[:, 1].max())
            ax.set_zlim(all_points[:, 2].min(), all_points[:, 2].max())

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if self.spatiotemporal_dim == 3:
            ax.set_zlabel("t")
        plt.axis("equal")

        if save_fig:
            plt.savefig(f"{folder}/scenario.{format}", format=format, dpi=300)

    # -----------------------------------------------------------
    # -------------------- ExtractSolution ----------------------

    def _extract_solution_raw(self, result):
        """
        Helper function to extract a sequence of spatio-temporal Bspline
        trajectories from the solution.
        """
        curves = []

        def get_outgoing_edge(vertex):
            edge = None
            phi = -1.0
            for e in self.gcs.Edges():
                if (e.u() == vertex) and result.GetSolution(e.phi()) > phi:
                    edge = e
                    phi = result.GetSolution(e.phi())
            return edge

        v = self.gcs_verts[self.start_vertex]
        while v != self.gcs_verts[self.end_vertex]:
            e = get_outgoing_edge(v)
            xu = result.GetSolution(e.xu())
            control_points = xu.reshape(self.order + 1, -1)
            basis = BsplineBasis(
                self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0, 1
            )
            curves.append(BsplineTrajectory(basis, control_points.T))
            v = e.v()

        return curves

    def ExtractSolution(self, result):
        """
        Extract a sequence of Bspline trajectories for space and time.

        Args:
            result: MathematicalProgramResult from calling SolveShortestPath.

        Returns:
            A tuple containing two lists: (spatial_trajectories, temporal_trajectories).
        """
        spatial_trajectories = []
        temporal_trajectories = []

        raw_trajectories = self._extract_solution_raw(result)

        for traj in raw_trajectories:
            control_points = np.array(traj.control_points()).T

            spatial_control_points = control_points[:, :-1]
            temporal_control_points = control_points[:, -1]

            # Create a Bspline for the spatial part
            spatial_trajectories.append(
                BsplineTrajectory(traj.basis(), spatial_control_points.T)
            )

            # Create a Bspline for the temporal part
            # The key is to reshape the temporal control points correctly
            # It needs to be a 2D array with shape (1, num_control_points)
            temporal_trajectories.append(
                BsplineTrajectory(traj.basis(), temporal_control_points.reshape(1, -1))
            )

        return (spatial_trajectories, temporal_trajectories)

    # -----------------------------------------------------------
    # -------------------- AnimateSolution ----------------------

    def AnimateSolution(
        self,
        result,
        show=True,
        save=False,
        folder="results",
        filename="ani_solution.gif",
    ):
        """
        Animate the solution in 2D or 3D.
        """
        assert self.spatiotemporal_dim in [
            2,
            3,
        ], "animation only supported in 2D and 3D"

        fig = plt.gcf()
        if self.spatiotemporal_dim == 3:
            if not hasattr(fig, "_ax3d"):
                ax = fig.add_subplot(111, projection="3d")
                fig._ax3d = ax
            else:
                ax = fig._ax3d
        else:
            ax = plt.gca()

        # Call the old function to get the full spatio-temporal trajectories
        s = self._extract_solution_raw(result)

        if self.spatiotemporal_dim == 2:
            q = ax.scatter(*s[0].value(0), color="blue", s=50, zorder=3)
        else:
            init = s[0].value(0).flatten()
            q = ax.scatter(init[0], init[1], init[2], color="blue", s=50)

        def animate(t):
            segment = floor(t)
            new_q = s[segment].value(t % 1).flatten()
            if self.spatiotemporal_dim == 2:
                q.set_offsets(new_q)
            else:
                q._offsets3d = (
                    np.array([new_q[0]]),
                    np.array([new_q[1]]),
                    np.array([new_q[2]]),
                )
            return q

        t = np.arange(0, len(s), 0.02)
        ani = animation.FuncAnimation(fig, animate, t, interval=50, blit=False)

        if save:
            print(f"Saving animation to {folder}/{filename}, this may take a minute...")
            ani.save(f"{folder}/{filename}", writer=animation.PillowWriter(fps=30))

        if show:
            plt.show()

        return ani

    # -----------------------------------------------------------
    # -------------------- PlotSolution -------------------------

    def PlotSolution(
        self,
        result,
        plot_control_points=True,
        plot_path=True,
        save_fig=False,
        format="png",
        folder="results",
    ):
        """
        Plot solution paths (2D or 3D).
        """
        fig = plt.gcf()

        if self.spatiotemporal_dim == 3:
            # create 3D axis if not already present
            if not hasattr(fig, "_ax3d"):
                ax = fig.add_subplot(111, projection="3d")
                fig._ax3d = ax  # store so we reuse the same axis
            else:
                ax = fig._ax3d
        else:
            ax = plt.gca()

        for edge in self.gcs.Edges():
            phi = result.GetSolution(edge.phi())
            xu = result.GetSolution(edge.xu())

            if phi > 0.0:
                control_points = xu.reshape(self.order + 1, -1)
                basis = BsplineBasis(
                    self.order + 1, self.order + 1, KnotVectorType.kClampedUniform, 0, 1
                )
                path = BsplineTrajectory(basis, control_points.T)

                if plot_control_points:
                    if self.spatiotemporal_dim == 2:
                        ax.plot(
                            control_points[:, 0],
                            control_points[:, 1],
                            "o--",
                            color="red",
                            alpha=phi,
                        )
                    else:
                        ax.plot(
                            control_points[:, 0],
                            control_points[:, 1],
                            control_points[:, 2],
                            "o--",
                            color="red",
                            alpha=phi,
                        )

                if plot_path:
                    curve = path.vector_values(np.linspace(0, 1))
                    if self.spatiotemporal_dim == 2:
                        ax.plot(
                            curve[0, :],
                            curve[1, :],
                            color="blue",
                            linewidth=2,
                            alpha=phi,
                        )
                    else:
                        ax.plot(
                            curve[0, :],
                            curve[1, :],
                            curve[2, :],
                            color="blue",
                            linewidth=2,
                            alpha=phi,
                        )
        if save_fig:
            plt.savefig(f"{folder}/solution.{format}", format=format, dpi=300)


# ============================================================
# -------------------- Example usage -------------------------
# ============================================================

if __name__ == "__main__":
    from stl_tool.stl import (
        FOp,
        GOp,
        BoxPredicate2d,
        Geq,
    )

    from stl_gcs.stl2gcs import STLTasks2GCS

    def toy_example():

        # ------- System definition -------------
        # Initial conditions
        # x0 = np.array([-0.4, -0.15])
        x0 = np.array([-0.3, 0.2])
        # State bounds
        dx = 1.5

        t0 = 0.0

        # --------- Predicate functions -------------
        # Goal positions
        pA = np.array([-0.3, 0.2])
        pB = np.array([0.35, 0.5])
        pC = np.array([0.35, -0.5])

        # Size of the square around the goal (half side length)
        dA = 0.45
        dB = 0.45
        dC = 0.45

        # # Test
        # print(np.linalg.norm(x0 - pA))

        # creates a box over the first three dimension  of the system (so on the positon).
        center = np.array([-0.0, 0.0])
        hX = BoxPredicate2d(size=dx, center=center, name="State bounds")
        h1 = BoxPredicate2d(size=dA, center=pA, name="Goal A")

        h2C = np.array([-10, 1])
        h2d = -2
        # h2H = Polyhedron(h2C, h2d)
        h2 = Geq(dims=[0, 1], state_dim=2, bound=h2d, name="Halfplane predicate")

        h3 = BoxPredicate2d(size=dB, center=pB, name="Goal B")
        h4 = BoxPredicate2d(size=dC, center=pC, name="Goal C")

        # --------- Formula definition -------------
        taG1 = 0.0
        tbG1 = 4.0
        formula1 = GOp(taG1, tbG1) >> h1

        taU2 = 5.0
        tbU2 = 8.0
        # formula2 = (GOp(0.0, taU2)) & (FOp(taU2, tbU2) >> h3)
        formula2 = FOp(taU2, tbU2) >> h3

        taF3 = 8.0
        tbF3 = 10.0
        formula3 = FOp(taF3, tbF3) >> h4

        formula = formula1 & formula2 & formula3

        # formula.show_graph()

        # plt.show()

        stl_tasks2gcs = STLTasks2GCS(formula=formula, xdim=2, r=0.05, x0=x0)

        # stl_tasks2gcs.plot3D_polytopes()

        graph = stl_tasks2gcs.graph
        vertices = graph.vertices
        edges = graph.edges
        vertex2polydict = stl_tasks2gcs.vertex2poly
        start_vertex = stl_tasks2gcs.start_vertex
        end_vertex = stl_tasks2gcs.end_vertex
        start_point = np.hstack((stl_tasks2gcs.x0, t0))

        # Create the Bspline GCS problem
        gcs = SpatioTemporalBsplineGraphOfConvexSets(
            vertices,
            edges,
            vertex2polydict,
            start_vertex,
            end_vertex,
            start_point,
            order=3,
            continuity=2,
        )

        # Add costs to the problem
        gcs.AddLengthCost(weight=1.0, norm="L2_squared")
        # gcs.AddDerivativeCost(degree=1, weight=10.0, norm="L2_squared")
        gcs.addTimeCost(weight=1.0)
        # gcs.addPathEnergyCost(weight=0.5)
        gcs.addDerivativeRegularization(order=2)
        gcs.addVelocityLimits(
            lower_bound=-0.5 * np.ones(2), upper_bound=0.5 * np.ones(2)
        )

        # Plot the scenario
        plt.figure(figsize=(8, 8))
        gcs.PlotScenario()
        plt.title("Scenario")
        # plt.xlabel("x")
        # plt.ylabel("y")
        # plt.xlim(-1, 7)
        # plt.ylim(-1, 7)
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
        # assert result.is_success()
        print(f"Optimal cost: {result.get_optimal_cost()}")

        # Plot the solution
        gcs.PlotSolution(result, plot_control_points=True, plot_path=True)
        plt.title("Optimal solution")
        plt.pause(0.1)  # pause to ensure the plot updates

        # plt.show()

        # Animate the solution
        gcs.AnimateSolution(result, show=True, save=False)

    toy_example()
