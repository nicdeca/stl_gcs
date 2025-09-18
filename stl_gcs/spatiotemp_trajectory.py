import numpy as np
from pydrake.all import (
    ConvexSet,
    VPolytope,
    HPolyhedron,
    Point,
)

from math import floor

from scipy.optimize import brentq


class SpatioTemporalBspline:
    """
    Trajectory composed of piecewise B-splines:
    - r_traj[i](s) : state-space B-spline for segment i
    - h_traj[i](s) : time mapping B-spline for segment i (outputs absolute time)

    Each segment is assumed to have domain s ∈ [0, 1].
    """

    def __init__(self, r_segments, h_segments):
        assert len(r_segments) == len(
            h_segments
        ), "r_segments and h_segments must have the same length"

        self.r_segments = r_segments
        self.h_segments = h_segments
        self.num_segments = len(r_segments)

        # Compute segment time boundaries
        self.segment_times = []
        for h in h_segments:
            t0 = h.value(0.0)[0].item()
            tf = h.value(1.0)[0].item()
            self.segment_times.append((t0, tf))

        self.t0 = self.segment_times[0][0]
        self.tf = self.segment_times[-1][1]

    def _find_segment(self, t, tol=1e-6):
        """
        Locate which segment contains the time t.
        """
        assert (
            self.t0 - tol <= t <= self.tf + tol
        ), f"t={t} outside trajectory time bounds [{self.t0}, {self.tf}]"

        for i, (t0, tf) in enumerate(self.segment_times):
            if t0 - tol <= t <= tf + tol:
                return i
        raise RuntimeError(f"Could not find segment for t={t}")

    def _find_s(self, h, t, tol=1e-8):
        """
        Invert h(s) = t for s ∈ [0,1] on one segment.
        """

        def f(s):
            return h.value(s).item() - float(t)

        if abs(f(0.0)) <= tol:
            return 0.0
        if abs(f(1.0)) <= tol:
            return 1.0
        return brentq(f, 0.0, 1.0, xtol=tol)

    def eval(self, t):
        """
        Evaluate reference state at global time t.
        """
        i = self._find_segment(t)
        s = self._find_s(self.h_segments[i], t)
        return self.r_segments[i].value(s).flatten()

    def deriv(self, t, order=1, time_derivative=True):
        """
        Evaluate derivative of reference trajectory at global time t.

        Parameters
        ----------
        order : int
            Derivative order.
        time_derivative : bool
            If True, apply chain rule to return derivative wrt real time.
            If False, return derivative wrt spline parameter s.
        """
        i = self._find_segment(t)
        r = self.r_segments[i]
        h = self.h_segments[i]

        s = self._find_s(h, t)

        if not time_derivative:
            return r.MakeDerivative(order).value(s).flatten()

        # Chain rule: dx/dt = (dr/ds) / (dh/ds)
        if order == 1:
            dr_ds = r.MakeDerivative(1).value(s).flatten()
            dh_ds = h.MakeDerivative(1).value(s).item()
            return dr_ds / dh_ds
        elif order == 2:
            # d²x/dt² = (d²r/ds² * (dh/ds) - dr/ds * d²h/ds²) / (dh/ds)^3
            dr_ds = r.MakeDerivative(1).value(s).flatten()
            d2r_ds2 = r.MakeDerivative(2).value(s).flatten()
            dh_ds = h.MakeDerivative(1).value(s).item()
            d2h_ds2 = h.MakeDerivative(2).value(s).item()
            return (d2r_ds2 * dh_ds - dr_ds * d2h_ds2) / (dh_ds**3)
        else:
            raise NotImplementedError("Higher order time derivatives not implemented")

    def time_span(self):
        return self.t0, self.tf

    def plot_vs_time(
        self,
        num_samples=200,
        order=0,
        time_derivative=True,
        show=True,
        ax=None,
        labels=None,
        save_fig=False,
        format="png",
        folder="results",
        filename=None,
    ):
        """
        Plot each spatial dimension of the trajectory or its derivatives as a function of time.

        Parameters
        ----------
        num_samples : int
            Number of time samples.
        order : int
            Derivative order.
            - 0 → plot trajectory
            - 1 → plot velocity
            - 2 → plot acceleration
        time_derivative : bool
            If True, compute derivatives w.r.t. real time (using chain rule).
        show : bool
            If True, call plt.show() at the end.
        ax : matplotlib.axes.Axes, optional
            If provided, plot into this axis. Otherwise, create a new figure.
        labels : list of str, optional
            Labels for each spatial dimension (e.g. ["x", "y", "z"]).
        save_fig : bool
            If True, save the figure.
        format : str
            Format for saving the figure (e.g. 'png', 'pdf').
        folder : str
            Destination folder for saved figure.
        filename : str, optional
            Name of the file to save (default is based on order).
        """
        import matplotlib.pyplot as plt

        t0, tf = self.time_span()
        ts = np.linspace(t0, tf, num_samples)

        # Evaluate trajectory or derivative
        if order == 0:
            xs = np.array([self.eval(t) for t in ts])
        else:
            xs = np.array(
                [
                    self.deriv(t, order=order, time_derivative=time_derivative)
                    for t in ts
                ]
            )

        dim = xs.shape[1]

        # Prepare axis
        if ax is None:
            fig, ax = plt.subplots()

        for d in range(dim):
            lbl = labels[d] if labels is not None else f"dim {d}"
            if order == 1:
                lbl += " (velocity)"
            elif order == 2:
                lbl += " (acceleration)"
            ax.plot(ts, xs[:, d], label=lbl)

        ax.set_xlabel("time [s]")
        ax.set_ylabel("value")
        ax.legend()
        ax.grid(True)

        # Save the plot
        if save_fig:
            if filename is None:
                if order == 0:
                    filename = "trajectory_vs_time"
                elif order == 1:
                    filename = "velocity_vs_time"
                elif order == 2:
                    filename = "acceleration_vs_time"
            plt.savefig(f"{folder}/{filename}.{format}")

        if show:
            plt.show()

    def plot_spatial(
        self,
        num_samples=200,
        show=True,
        ax=None,
        plot_control_points=False,
        labels=("x", "y"),
        save_fig=False,
        format="png",
        folder="results",
        filename="spatial_trajectory",
        **kwargs,
    ):
        """
        Plot the spatial trajectory in the 2D workspace.

        Parameters
        ----------
        num_samples : int
            Number of time samples.
        show : bool
            If True, call plt.show() at the end.
        ax : matplotlib.axes.Axes, optional
            If provided, plot into this axis. Otherwise, use plt.gca().
        plot_control_points : bool
            If True, also plot the control points of each r_segment.
        labels : tuple of str
            Labels for spatial dimensions, e.g. ("x", "y").
        save_fig : bool
            If True, save the figure.
        format : str
            Format for saving the figure (e.g. 'png', 'pdf').
        folder : str
            Destination folder for saved figure.
        filename : str
            Name of the file to save.
        kwargs : dict
            Extra arguments passed to ax.plot().
        """
        import matplotlib.pyplot as plt

        t0, tf = self.time_span()
        ts = np.linspace(t0, tf, num_samples)

        # Evaluate spatial trajectory
        xs = np.array([self.eval(t) for t in ts])

        # Ensure 2D
        if xs.shape[1] < 2:
            raise ValueError("Spatial trajectory must have at least 2 dimensions")

        # Get axis
        if ax is None:
            ax = plt.gca()

        # Plot trajectory
        ax.plot(xs[:, 0], xs[:, 1], label="trajectory", **kwargs)

        # Plot control points if requested
        if plot_control_points:
            for r in self.r_segments:
                # cps = r.control_points()
                cps = np.array([cp.flatten() for cp in r.control_points()])
                ax.plot(cps[:, 0], cps[:, 1], "o--", alpha=0.5, label="control points")

        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.legend()
        ax.grid(True)

        # Save the plot
        if save_fig:
            plt.savefig(f"{folder}/{filename}.{format}")

        if show:
            plt.show()


class PiecewisePolytopes:
    """
    Represents a sequence of polytopes over a time domain.
    """

    def __init__(self, time_segments, polytopes):
        """
        Args:
            time_segments (list of tuples): A list of (t_start, t_end) for each segment.
            polytopes (list of ConvexSet): A list of ConvexSet objects corresponding
                                          to each time segment.
        """
        assert len(time_segments) == len(
            polytopes
        ), "The number of time segments must match the number of polytopes."

        self.time_segments = time_segments
        self.polytopes = polytopes
        self.num_segments = len(polytopes)

    def _find_segment_index(self, t, tol=1e-6):
        """
        Finds the index of the segment active at time t.
        """
        for i, (t_start, t_end) in enumerate(self.time_segments):
            if t_start - tol <= t <= t_end + tol:
                return i

        raise ValueError(f"Time {t} is outside the defined time segments.")

    def get_active_polytope(self, t):
        """
        Returns the active polytope at a specific time.

        Args:
            t (float): The time to query.

        Returns:
            The ConvexSet object for the active polytope.
        """
        index = self._find_segment_index(t)
        return self.polytopes[index]

    def get_time_section(self, t):
        """
        Evaluates the spatial cross-section of the active polytope at a specified time t.

        This method is specific to your problem where the last dimension is time.
        The resulting section is a new polytope with one fewer dimension.

        Args:
            t (float): The time to get the cross-section.

        Returns:
            A new HPolyhedron representing the spatial cross-section.
        """
        index = self._find_segment_index(t)
        poly = self.polytopes[index]

        if not isinstance(poly, HPolyhedron):
            # Convert to H-representation if needed, as it's easier to slice
            # This handles both VPolytope and other ConvexSet types
            poly = HPolyhedron(poly)

        # The H-representation is Ax <= b. We want to fix the last dimension (time)
        # to the value 't'.
        # The last column of A corresponds to the time dimension.
        # This adds an equality constraint x_time = t.
        A_orig = poly.A()
        b_orig = poly.b()

        # Add the time constraint as two inequalities: x_time <= t and -x_time <= -t
        A_time_eq = np.zeros((2, A_orig.shape[1]))
        b_time_eq = np.zeros(2)

        A_time_eq[0, -1] = 1.0
        A_time_eq[1, -1] = -1.0
        b_time_eq[0] = t
        b_time_eq[1] = -t

        # Combine the original constraints with the new time constraints
        A_section = np.vstack([A_orig, A_time_eq])
        b_section = np.concatenate([b_orig, b_time_eq])

        # Create a new HPolyhedron from the combined constraints
        return HPolyhedron(A_section, b_section)


if __name__ == "__main__":
    from stl_tool.stl import (
        FOp,
        GOp,
        BoxPredicate2d,
        Geq,
    )

    from stl_gcs.stl2gcs import STLTasks2GCS
    from stl_gcs.spatiotemp_gcs import SpatioTemporalBsplineGraphOfConvexSets
    import matplotlib.pyplot as plt

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
        pD = np.array([-0.4, -0.4])
        pE = np.array([0.1, -0.3])
        pF = np.array([0.2, 0.4])
        pG = np.array([0.5, 0.0])
        pH = np.array([-0.6, 0.3])

        # Sizes of the square around the goal (half side length)
        dA, dB, dC, dD, dE, dF, dG, dH = 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25
        # dA, dB, dC, dD, dE, dF, dG, dH = 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.4, 0.4

        # Predicates
        hA = BoxPredicate2d(size=dA, center=pA, name="Goal A")
        hB = BoxPredicate2d(size=dB, center=pB, name="Goal B")
        hC = BoxPredicate2d(size=dC, center=pC, name="Goal C")
        hD = BoxPredicate2d(size=dD, center=pD, name="Goal D")
        hE = BoxPredicate2d(size=dE, center=pE, name="Goal E")
        hF = BoxPredicate2d(size=dF, center=pF, name="Goal F")
        hG = BoxPredicate2d(size=dG, center=pG, name="Goal G")
        hH = BoxPredicate2d(size=dH, center=pH, name="Goal H")

        # # Temporal operators
        # f1 = GOp(0.0, 3.0) >> hA  # Always at A
        # f2 = FOp(4.0, 6.0) >> hB  # Eventually B
        # f3 = FOp(5.0, 8.0) >> hC  # Eventually C
        # f4 = GOp(2.0, 5.0) >> hD  # Always D
        # f5 = FOp(6.0, 9.0) >> hE  # Eventually E
        # f6 = FOp(1.0, 4.0) >> hF  # Eventually F
        # f7 = GOp(3.0, 7.0) >> hG  # Always at G
        # f8 = FOp(2.0, 6.0) >> hH  # Eventually H

        # # Conjunctions (single temporal operators per child)
        # conj1 = f1 & f2  # Plan 1: stay at A, eventually B
        # conj2 = f4 & f3  # Plan 2: eventually C, always D
        # conj3 = f6 & f5  # Plan 3: eventually E, eventually F

        # conj4 = f1 & f2 & f7  # Stay at A, eventually B, always G
        # conj5 = f3 & f5 & f8  # Eventually C, eventually E, eventually H
        # conj6 = f4 & f6 & f7 & f8  # Always D, eventually F, always G, eventually H
        # Plan 1
        conj1 = (GOp(0.0, 3.0) >> hA) & (FOp(4.0, 6.0) >> hB)

        # Plan 2
        conj2 = (GOp(2.0, 5.0) >> hD) & (FOp(5.0, 8.0) >> hC)

        # Plan 3
        conj3 = (FOp(1.0, 4.0) >> hF) & (FOp(6.0, 9.0) >> hE)

        # Plan 4
        conj4 = (GOp(0.0, 3.0) >> hA) & (FOp(4.0, 6.0) >> hB) & (GOp(7.0, 9.0) >> hG)

        # Plan 5
        conj5 = (
            (FOp(5.0, 8.0) >> hC) & (FOp(10.0, 11.0) >> hE) & (FOp(12.0, 16.0) >> hH)
        )

        # Plan 6
        conj6 = (
            (GOp(2.0, 3.0) >> hD)
            & (FOp(4.0, 4.5) >> hF)
            & (GOp(5.0, 7.0) >> hG)
            & (FOp(10.0, 11.0) >> hH)
        )

        formula = conj1 | conj2 | conj3 | conj4 | conj5 | conj6

        # Disjunction of the three conjunctions (three alternative plans)
        formula = conj1 | conj2 | conj3 | conj4 | conj5 | conj6

        stl_tasks2gcs = STLTasks2GCS(formula=formula, xdim=2, r=0.05, x0=x0)

        stl_tasks2gcs.plot3D_polytopes()

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
            degree=3,
            continuity=2,
        )

        # Add costs to the problem
        gcs.AddLengthCost(weight=1.0, norm="L2_squared")
        # gcs.AddDerivativeCost(degree=1, weight=10.0, norm="L2_squared")
        gcs.addTimeCost(weight=1.0)
        # gcs.addPathEnergyCost(weight=0.5)
        gcs.addDerivativeRegularization(deriv_order=2)
        # gcs.addVelocityLimits(
        #     lower_bound=-0.5 * np.ones(2), upper_bound=0.5 * np.ones(2)
        # )

        # Plot the scenario
        plt.figure(figsize=(8, 8))
        gcs.PlotScenario(save_fig=False, format="png", folder="results")
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
        gcs.PlotSolution(
            result,
            plot_control_points=True,
            plot_path=True,
            save_fig=False,
            format="png",
            folder="results",
        )
        plt.title("Optimal solution")
        plt.pause(0.1)  # pause to ensure the plot updates

        # plt.show()

        # Animate the solution
        # gcs.AnimateSolution(result, show=True, save=False, filename=None)

        spatial_trajectories, temporal_trajectories = gcs.ExtractSolution(result)

        # Instantiate the combined trajectory manager
        combined_trajectory = SpatioTemporalBspline(
            spatial_trajectories, temporal_trajectories
        )

        # Use the methods to evaluate the trajectory at a specific time
        # Example: Evaluate the state at the midpoint of the total time span
        t_mid = (combined_trajectory.t0 + combined_trajectory.tf) / 2
        state_at_midpoint = combined_trajectory.eval(t_mid)

        print(f"Total time span: {combined_trajectory.time_span()}")
        print(f"State at t={t_mid:.2f}: {state_at_midpoint}")

        velocity_at_midpoint = combined_trajectory.deriv(t_mid, order=1)
        acceleration_at_midpoint = combined_trajectory.deriv(t_mid, order=2)

        print(f"Velocity at t={t_mid:.2f}: {velocity_at_midpoint}")
        print(f"Acceleration at t={t_mid:.2f}: {acceleration_at_midpoint}")

        # Plot the trajectory components over time
        combined_trajectory.plot_vs_time(labels=["x", "y"], save_fig=False)

        # Plot velocity
        combined_trajectory.plot_vs_time(labels=["vx", "vy"], order=1, save_fig=False)

    toy_example()
