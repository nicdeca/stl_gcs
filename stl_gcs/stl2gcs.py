import numpy as np

from typing import Optional, Union, Tuple
import cvxpy as cp

from stl_tool.polyhedron import Polyhedron

from stl_tool.stl import (
    Formula,
    AndOperator,
    OrOperator,
    UOp,
    FOp,
    GOp,
    get_formula_type_and_predicate_node,
    TimeInterval,
    BoxPredicate2d,
    Geq,
)

from pydrake.all import HPolyhedron

from stl_gcs.graph import Graph


class GCSTaskScheduler:
    """
    This class takes as input a formula and the state space dimension and creates a time schedule for the tasks
    in the form of a Graph of Convex Sets. Notice that formula already integrates a tree of static and temporal
    operators and predicates, however specific rules apply to construct a graph suitable for the finding
    the optimal path on a graph of convex sets.
    Notice that, here, there is no reference to the actual gcs class (which is the solver of the problem), here,
    we just create the graph of convex sets structure, the polytopes and the time intervals associated to each task.
    """

    def __init__(self, formula: Formula, xdim: int, r: float, x0: np.ndarray) -> None:
        """
        :param formula: The STL formula to be satisfied.
        :param xdim: The dimension of the state space.
        :param r: The minimum desired robustness of the formula.
        :param x0: The initial state of the system.
        TODO: Add until operator, FG and GF operators.
        """
        self.formula: Formula = formula
        self.xdim: int = xdim
        self.r: float = r

        self.x0: np.ndarray = x0

        # Define a graph for the GCS formulation. Vertices and edges are initially the empty list
        vertices = []
        edges = []
        self.graph = Graph(vertices, edges)
        self.start_vertex = None
        self.end_vertex = None

        self.vertex2poly: dict = {}  # map each vertex to a time-varying polytope

        self.build_stl_gcs()

    def build_stl_gcs(self):
        """
        Iterate through the formula tree and associate to each task a time-varying polytope.
        Build the graph of convex sets from the given formula.
        """

        # 1) Get the list of tasks from the formula
        is_a_conjunction = False

        # TODO: Add the OR operator
        # Check that the provided formula is within the fragment of allowable formulas.
        if isinstance(self.formula.root, OrOperator):
            # 1) the root operator can be an or, but it not implemented for now.
            raise NotImplementedError(
                "OrOperator is not implemented yet. Wait for it. it is coming soon."
            )
        elif isinstance(self.formula.root, AndOperator):
            # then it is a conjunction of single formulas
            is_a_conjunction = True
        else:
            # otherwise it is a single formula
            pass

        # subdivide in sumbformulas
        potential_varphi_tasks: list[Formula] = []
        if is_a_conjunction:
            for child_node in self.formula.root.children:
                # take all the children nodes and check that the remaining formulas are in the predicate
                varphi = Formula(root=child_node)
                potential_varphi_tasks += [varphi]

        else:
            potential_varphi_tasks = [self.formula]

        # TODO: When adding ORs, all the next code should be in a loop for each disjunctive branch

        prev_polyID = None
        prev_task_polytope = None
        t_start_next_task = 0.0

        for varphi in potential_varphi_tasks:
            # take all the children nodes and check that the remaining formulas are in the predicate

            varphi_type, predicate_node = get_formula_type_and_predicate_node(
                formula=varphi
            )

            root: Union[FOp, UOp, GOp] = varphi.root  # Temporal operator of the fomula.
            # Time interval of the formula.
            time_interval: TimeInterval = varphi.root.interval
            # Output matrix of the predicate node.      # TODO: The whole configuration should be constrained by the polytope so C should always be identity
            C: np.ndarray = predicate_node.output_matrix

            # Bring polytope to suitable dimension
            curr_task_polytope = Polyhedron(
                A=predicate_node.polytope.A @ C, b=predicate_node.polytope.b
            )

            # Define the initial polytope at time t0 so that it contains the state at time 0 if t0=0
            # and it contains the the set at time t->t0^- otherwise
            t_start_task = t_start_next_task
            if varphi_type == "G":
                # Define the two polytopes corresponding to the G operator
                t_end = time_interval.a
                # Set the initial time for the next polytope
                t_start_next_task = time_interval.b

            elif varphi_type == "F":
                t_end = time_interval.b
                # Set the initial time for the next polytope
                t_start_next_task = time_interval.a

            else:
                raise NotImplementedError(
                    "Only G and F operators are implemented for now."
                )

            tv_poly = None

            if t_start_task == t_end:
                # This can typically happen for an always task which is active starting from t_start
                # Two cases are possible:
                # 1) The second polytope already contains the first one, so gamma = 0
                # 2) The second polytope does not contain the first one, so the task is infeasible
                # tv_poly is not necessary
                if prev_polyID is None:
                    # first task, check if x0 satisfies the predicate at time t_start
                    self.check_feasibility_x0_tstart(curr_task_polytope)
                else:
                    self.check_feasibility_prevset_tstart(
                        curr_task_polytope, prev_task_polytope
                    )
            else:
                # A time-varying polytope is only needed when the task starts strictly before the time at which the predicate
                # function has to be satisfied i.e. interval.a for an always operator or interval.b for an eventually operator
                tv_poly = self.build_tv_polytope(
                    prev_task_polytope, curr_task_polytope, t_start_task, t_end
                )

            tc_poly = self.build_tc_polytope(
                curr_task_polytope, time_interval.a, time_interval.b
            )

            polyID1, polyID2 = self.append_to_graph(
                predicate_node.node_id, prev_polyID, tv_poly, tc_poly
            )

            prev_task_polytope = curr_task_polytope
            if self.start_vertex is None:
                self.start_vertex = polyID1
            prev_polyID = polyID2

        # TODO: Set end vertex (if multiple disjunctive branches, you might have multiple end vertices, just add a dummy vertex and connect them all to it)

        pass

    def append_to_graph(
        self,
        formula_id: int,
        prev_polyID: int | None,
        tv_poly: HPolyhedron | None,
        tc_poly: HPolyhedron,
    ) -> tuple[int, int]:
        """
        Append two vertices to the graph, connect them and record their polytopes.

        - polyID1 = time-varying vertex (active during [t_start, t_end])
        - polyID2 = time-constant vertex (predicate extended in time [a,b])

        If prev_polyID is provided, connect prev_polyID -> polyID1.
        """

        if tv_poly is not None:
            # Create an ID for the vertex associated to each of the two polytopes associated to the task
            # first polytope, the time-varying one (in [t_start, t_end])
            polyID1 = 2 * formula_id
            self.graph.add_vertex(polyID1)
            # second polytope, the constant one corresponding to the predicate function (in [a,b])
            polyID2 = 2 * formula_id + 1
            self.graph.add_vertex(polyID2)
            # connect the time-varying vertex to the time-constant (predicate-extended) vertex
            self.graph.add_edge((polyID1, polyID2))
            self.vertex2poly[polyID1] = tv_poly  # the first polytope
        else:
            # only one polytope is needed (degenerate case when t_start == t_end)
            polyID1 = 2 * formula_id + 1
            self.graph.add_vertex(polyID1)
            polyID2 = polyID1

        if prev_polyID is not None:
            #  connect previous task to the current time-varying polytope
            self.graph.add_edge((prev_polyID, polyID1))

        self.vertex2poly[polyID2] = tc_poly  # the second polytope

        return polyID1, polyID2

    def build_tc_polytope(
        self, curr_poly: Polyhedron, t_start: float, t_end: float
    ) -> HPolyhedron:
        """
        Build the time-constant polytope (cartesian product with time interval).
        Standard H-form for variables [x (n dims); t (1 dim)]:
            A_tc * [x; t] <= b_tc
        where the first block enforces Ax <= b (shifted by robustness r)
        and the last two rows enforce t_start <= t <= t_end:
            -t <= -t_start    ->  -1 * t <= -t_start
             t <=  t_end     ->   1 * t <=  t_end
        """
        cons_dim = curr_poly.A.shape[0]

        # Extend the inequality to consider time, shape (cons_dim, n+1)
        Atop = np.hstack([curr_poly.A, np.zeros((cons_dim, 1))])

        # time bounds rows (two inequalities)
        At_time = np.block(
            [[np.zeros((2, curr_poly.A.shape[1])), np.array([[-1.0], [1.0]])]]
        )

        Atc = np.vstack([Atop, At_time])

        btc = np.concatenate(
            [(curr_poly.b - self.r).flatten(), np.array([-t_start, t_end])]
        )

        # This time use pydrake Hpolyhedrons for compatibility with the GCS library
        tc_poly = HPolyhedron(Atc, btc)
        return tc_poly

    def build_tv_polytope(
        self,
        prev_poly: Polyhedron | None,
        curr_poly: Polyhedron,
        t_start: float,
        t_end: float,
    ) -> HPolyhedron:
        """
        Build a time-varying polytope as an H-polyhedron in variables [x; t],
        parameterized by minimal gamma computed to ensure that curr_poly contains
        prev_poly at time t_start.
        Standard H-form for variables [x (n dims); t (1 dim)]:
            A_tv * [x; t] <= b_tv
        where the first block enforces:
        Ax <= b + ( gamma * (1 - t/t_end) - r ) * ones(cons_dim)
        and the last two rows enforce t_start <= t <= t_end:
            -t <= -t_start    ->  -1 * t <= -t_start
             t <=  t_end     ->   1 * t <=  t_end
        Note that the time-varying polytope is only defined in the interval [t_start, t_end].
        """

        if prev_poly is None:
            gamma = self.gamma_for_initial_state(curr_poly, t_start, t_end)
        else:
            # Compute the minimum slope of the time-varying polytope such that
            # prev_poly(t_k) ⊆ curr_poly(t_k, gamma)
            gamma = self.minimal_gamma(prev_poly, curr_poly, t_start, t_end)

        cons_dim = curr_poly.A.shape[0]

        # A_tv = [-A, (gamma/alpha) * 1]
        A_left = curr_poly.A  # shape (cons_dim, n)
        A_time_coeff = (gamma / t_end) * np.ones((cons_dim, 1))  # shape (cons_dim, 1)
        Atop = np.hstack([A_left, A_time_coeff])  # (cons_dim, n+1)

        # time bounds rows (two inequalities)
        At_time = np.block(
            [[np.zeros((2, curr_poly.A.shape[1])), np.array([[-1.0], [1.0]])]]
        )

        Atv = np.vstack([Atop, At_time])

        # b_tv = b + gamma - r for the inequality block; then time bounds as usual
        b_upper = (curr_poly.b + gamma - self.r).flatten()
        b_time = np.array([-t_start, t_end])
        btv = np.concatenate([b_upper, b_time])

        # This time use pydrake Hpolyhedrons for compatibility with the GCS library
        tv_poly = HPolyhedron(Atv, btv)
        return tv_poly

    def gamma_for_initial_state(
        self, poly: Polyhedron, t_start: float, t_end: float
    ) -> float:
        """
        Compute minimal gamma so that the first time-varying polytope contains x0 at t_start.

        b(x0, t_start) = A @ x0 <= b - gamma/t_end * t_start + gamma - r
        """

        A = poly.A
        b = poly.b.flatten()

        vals = A @ self.x0 - b
        max_val = np.max(vals)
        # if max_val <= 0, then x0 is already inside the polytope
        # if max_val > 0, we need to increase gamma

        denom = 1.0 - t_start / t_end
        if denom <= 0:
            raise ValueError(
                "Invalid time parameters: denom = 1 - t_start/t_end <= 0. Check t_start/t_end ordering."
            )

        if max_val <= -self.r:
            return 0.0
        else:
            return (self.r + max_val) / denom

        # gamma0 = max(0.0, self.r + max_val) / denom
        # return gamma0

    def minimal_gamma(
        self,
        prev_poly: Polyhedron,
        next_poly: Polyhedron,
        t_start: float,
        t_end: float,
    ):
        """
        Compute minimal gamma such that prev_poly(t_start) ⊆ next_poly(t_start,gamma).
        The containment condition reduces to, for each inequality i of next_poly:
            vi = max_{x in prev_poly} ( a_i^T x ) <= b_i + gamma - r - (gamma/t_end) * t_start
        Rearranged to isolate gamma leads to:
            gamma >= (v_i - (b_i - r)) / (1 - t_start/t_end) for all i
        """

        A1, b1 = prev_poly.A, prev_poly.b.flatten()
        A2, b2 = next_poly.A, next_poly.b.flatten()

        denom = 1.0 - (t_start / t_end)
        if denom <= 0:
            raise ValueError(
                "Invalid time parameters: denom = 1 - t_start/t_end <= 0. Check t_start/t_end ordering."
            )

        gammas = []
        x = cp.Variable(self.xdim)

        # Choose solver: prefer MOSEK if available, otherwise use SCS
        solver_choice = None
        # try MOSEK first (most robust)
        solver_choice = cp.MOSEK

        for i in range(A2.shape[0]):
            obj = cp.Maximize(A2[i, :] @ x)
            cons = [A1 @ x <= b1]
            prob = cp.Problem(obj, cons)
            try:
                prob.solve(solver=solver_choice, warm_start=True, verbose=False)
            except Exception:
                # fallback to a different solver
                prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise RuntimeError(
                    f"Containment LP failed for constraint {i}: status {prob.status}"
                )

            v_star = float(prob.value)  # optimal value of max(a_i^T x)
            # note: earlier we derived numer = v_star - (b2[i] - r)
            numer = v_star - (b2[i] - self.r)
            gamma_i = numer / denom
            gammas.append(gamma_i)

        gamma_required = float(np.max(gammas))
        # numerical safety: don't return negative gamma
        if gamma_required < 0:
            gamma_required = 0.0

        return gamma_required

    def minimal_gamma_dual(
        self, prev_poly: Polyhedron, next_poly: Polyhedron, t_start: float, t_end: float
    ) -> float:
        """
        Compute minimal gamma such that prev_poly(t_start) ⊆ next_poly(t_start, gamma),
        using the dual LP formulation of support functions. This approach can be more
        efficient when A2 has many rows and A1 has few rows.
        """

        A1, b1 = prev_poly.A, prev_poly.b.flatten()
        A2, b2 = next_poly.A, next_poly.b.flatten()

        denom = 1.0 - (t_start / t_end)
        if denom <= 0:
            raise ValueError("Invalid time parameters: denom <= 0.")

        gammas = []

        # dual variable lambda (size = number of constraints in prev_poly)
        lam = cp.Variable(A1.shape[0], nonneg=True)

        for i in range(A2.shape[0]):
            a2_row = A2[i, :]

            # constraint A1^T lam = a2_row
            cons = [A1.T @ lam == a2_row]

            obj = cp.Minimize(b1 @ lam)  # dual objective

            prob = cp.Problem(obj, cons)
            try:
                prob.solve(solver=cp.MOSEK, verbose=False)
            except Exception:
                prob.solve(solver=cp.SCS, verbose=False)

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise RuntimeError(
                    f"Dual LP failed for inequality {i}: status {prob.status}"
                )

            v_star = float(prob.value)  # support function value
            numer = v_star - (b2[i] - self.r)
            gamma_i = numer / denom
            gammas.append(gamma_i)

        gamma_required = float(np.max(gammas))
        return max(0.0, gamma_required)  # safety

    def check_feasibility_x0_tstart(self, poly: Polyhedron) -> float:
        """
        Check if the initial state x0 satisfies the first task's predicate at time t_start.
        """
        # This can typically happen for an always task which is active starting from t_start
        # Two cases are possible:
        # 1) A @ self.x0 - b <= 0, i.e. the predicate function is already satisfied at time t_start
        # Then we can set gamma = 0
        # 2) A @ self.x0 - b > 0, i.e. the predicate function is not satisfied at time t_start
        # In this case, the task is infeasible
        # A = poly.A
        # b = poly.b.flatten()

        # vals = A @ self.x0 - b
        # max_val = np.max(vals)
        # if max_val > 0:
        if self.x0 not in poly:
            raise RuntimeError(
                "Initial state x0 does not satisfy the first task's predicate at time t_start, and the predicate"
                "function is required to hold at t_start. Task is infeasible."
            )

    def check_feasibility_prevset_tstart(
        self, curr_poly: Polyhedron, prev_poly: Polyhedron
    ) -> float:
        """
        Check if the previous state satisfies the current task's predicate at time t_start.
        """
        # This can typically happen for an always task which is active starting from t_start
        # Two cases are possible:
        # 1) The second polytope already contains the first one, so gamma = 0
        # 2) The second polytope does not contain the first one, so the task is infeasible
        x = cp.Variable(self.xdim)

        A1, b1 = prev_poly.A, prev_poly.b.flatten()
        A2, b2 = curr_poly.A, curr_poly.b.flatten()

        for i in range(A2.shape[0]):
            obj = cp.Maximize(A2[i, :] @ x)
            cons = [A1 @ x <= b1]
            prob = cp.Problem(obj, cons)
            prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

            if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                raise RuntimeError(
                    f"LP failed while checking containment for constraint {i}"
                )

            v_star = prob.value
            if v_star > b2[i] - self.r + 1e-9:  # small tolerance
                return False  # containment fails

        return True  # all constraints satisfied

    def plot3D_polytopes(self, show=True, cmap_name="tab20"):
        """
        Plot the polytopes in 3D (x1, x2, t) directly as convex hulls.
        Expects self.vertex2poly to map vertex IDs -> HPolyhedron objects.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as cm
        from scipy.spatial import ConvexHull
        from pydrake.all import VPolytope

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        cmap = cm.get_cmap(cmap_name)

        for vid, poly in self.vertex2poly.items():

            # Check if empty
            if poly.IsEmpty():
                print("Polytope is empty!")
                continue
            # Convert HPolyhedron to vertices using Drake
            vpoly = VPolytope(poly)
            verts = np.array(vpoly.vertices().T)
            if verts.shape[0] < 4:
                raise RuntimeError(
                    "Polytope has less than 4 vertices, cannot plot 3D hull."
                )

            hull = ConvexHull(verts)
            faces = hull.simplices
            poly3d = [verts[face] for face in faces]

            facecolor = cmap((vid * 37) % cmap.N)
            coll = Poly3DCollection(
                poly3d, alpha=0.35, facecolor=facecolor, edgecolor="k"
            )
            ax.add_collection3d(coll)

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("t")
        ax.view_init(elev=25, azim=60)
        ax.grid(True)
        plt.tight_layout()
        if show:
            plt.show()
        return fig, ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def toy_example():

        # ------- System definition -------------
        # Initial conditions
        # x0 = np.array([-0.4, -0.15])
        x0 = np.array([-0.3, 0.2])
        # State bounds
        dx = 1.5

        # --------- Predicate functions -------------
        # Goal positions
        pA = np.array([-0.3, 0.2])
        pB = np.array([0.35, 0.5])
        pC = np.array([0.35, -0.5])

        # Size of the square around the goal (half side length)
        dA = 0.45
        dB = 0.45
        dC = 0.45

        # Test
        print(np.linalg.norm(x0 - pA))

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

        formula.show_graph()

        # plt.show()

        scheduler = GCSTaskScheduler(formula=formula, xdim=2, r=0.05, x0=x0)

        scheduler.plot3D_polytopes()

    toy_example()
