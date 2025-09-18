import numpy as np

from stl_tool.stl import (
    FOp,
    GOp,
    BoxPredicate2d,
    Geq,
)
from stl_tool.environment import Map
from stl_tool.polyhedron import Box2d

from stl_gcs.spatiotemp_trajectory import SpatioTemporalBspline
from stl_gcs.stl2gcs import STLTasks2GCS
from stl_gcs.spatiotemp_gcs import SpatioTemporalBsplineGraphOfConvexSets
import matplotlib.pyplot as plt

if __name__ == "__main__":

    def toy_example():

        # ------- System definition -------------
        # Initial conditions
        scaling = 3
        # x0 = np.array([-0.4, -0.15])
        x0 = scaling * np.array([-0.3, 0.2])
        # State bounds
        dx = 1.5

        t0 = 0.0

        # --------- Predicate functions -------------
        # Goal positions
        pA = scaling * np.array([-0.3, 0.2])
        pB = scaling * np.array([0.35, 0.5])
        pC = scaling * np.array([0.35, -0.5])
        pD = scaling * np.array([-0.4, -0.4])
        pE = scaling * np.array([0.1, -0.3])
        pF = scaling * np.array([0.2, 0.4])
        pG = scaling * np.array([0.5, 0.0])
        pH = scaling * np.array([-0.6, 0.3])

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
        conj1 = (
            (GOp(0.0, 3.0) >> hA)
            & (FOp(4.0, 6.0) >> hB)
            & (GOp(6.0, 9.0) >> hG)
            & (FOp(10.0, 12.0) >> hH)
            & (FOp(12.5, 16.0) >> hH)
            & (GOp(17.0, 18.0) >> hD)
        )

        # Plan 2
        conj2 = (
            (GOp(2.0, 5.0) >> hD)
            & (FOp(5.0, 8.0) >> hC)
            & (GOp(9.0, 10.0) >> hE)
            & (FOp(10.0, 12.0) >> hF)
            & (FOp(14.0, 16.0) >> hB)
            & (GOp(17.0, 19.0) >> hG)
        )

        # Plan 3
        conj3 = (
            (FOp(1.0, 4.0) >> hF)
            & (FOp(6.0, 9.0) >> hE)
            & (FOp(10.0, 11.5) >> hF)
            & (GOp(12.0, 14.0) >> hG)
            & (FOp(15.0, 17.0) >> hH)
        )

        # Plan 4
        conj4 = (
            (GOp(0.0, 3.0) >> hA)
            & (FOp(4.0, 6.0) >> hB)
            & (GOp(7.0, 9.0) >> hG)
            & (GOp(10.0, 13.0) >> hA)
            & (FOp(14.0, 16.0) >> hB)
        )

        # Plan 5
        conj5 = (
            (FOp(5.0, 8.0) >> hC)
            & (FOp(10.0, 11.0) >> hE)
            & (FOp(12.0, 16.0) >> hH)
            & (GOp(17.0, 18.0) >> hD)
            & (FOp(18.0, 20.0) >> hB)
        )

        # Plan 6
        conj6 = (
            (GOp(2.0, 3.0) >> hD)
            & (FOp(4.0, 4.5) >> hF)
            & (GOp(5.0, 7.0) >> hG)
            & (FOp(10.0, 11.0) >> hH)
            & (GOp(12.0, 15.0) >> hD)
            & (FOp(15.0, 18.0) >> hC)
        )

        # Disjunction of the three conjunctions (three alternative plans)
        formula = conj1 | conj2 | conj3 | conj4 | conj5 | conj6

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
            degree=3,
            continuity=2,
        )

        # Add costs to the problem
        gcs.AddLengthCost(weight=1.0, norm="L2_squared")
        # gcs.AddDerivativeCost(degree=1, weight=10.0, norm="L2_squared")
        gcs.addTimeCost(weight=1.0)
        # gcs.addPathEnergyCost(weight=0.5)
        gcs.addDerivativeRegularization(deriv_order=2)
        gcs.addVelocityLimits(
            lower_bound=-2.0 * np.ones(2), upper_bound=2.0 * np.ones(2)
        )

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
            convex_refinement=True,
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

        workspace = Box2d(
            x=0, y=0, size=5
        )  # square 10x10 (this is a 2d workspace, so the system it refers to must be 2d)
        map = Map(
            workspace=workspace
        )  # the map object contains the workpace, but it also contains the obstacles of your environment.

        map.draw()

        fig, ax = map.draw_formula_predicate(formula=formula, alpha=0.2)

        combined_trajectory.plot_spatial(
            num_samples=300,
            plot_control_points=True,
            color="blue",
            linewidth=2,
        )

        plt.show()

    toy_example()
