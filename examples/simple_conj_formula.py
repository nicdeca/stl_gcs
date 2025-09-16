import numpy as np
import matplotlib.pyplot as plt

from stl_tool.stl import (
    Formula,
    AndOperator,
    OrOperator,
    UOp,
    FOp,
    GOp,
    BoxPredicate2d,
    Geq,
)

from stl_gcs.stl2gcs import STLTasks2GCS
from stl_gcs.spatiotemp_gcs import SpatioTemporalBsplineGraphOfConvexSets
from stl_gcs.bspline_gcs_notime import BsplineGraphOfConvexSetsNoTime


if __name__ == "__main__":

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
        # gcs = SpatioTemporalBsplineGraphOfConvexSets(
        #     vertices,
        #     edges,
        #     vertex2polydict,
        #     start_vertex,
        #     end_vertex,
        #     start_point,
        #     order=3,
        #     continuity=2,
        # )
        gcs = BsplineGraphOfConvexSetsNoTime(
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

        # Animate the solution
        gcs.AnimateSolution(result, show=True, save=False, filename=None)

    toy_example()
