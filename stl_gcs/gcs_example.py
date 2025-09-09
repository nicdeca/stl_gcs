import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from pydrake.all import (
    AffineBall,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    Hyperellipsoid,
    MathematicalProgram,
    Point,
    Solve,
    VPolytope,
    GcsGraphvizOptions,
)

import graphviz


# -------------------------------------------------------------------
def Plot2dGraphOfConvexSets(gcs, ax):
    # Generate points on the unit circle for plotting ellipsoids
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_points = np.vstack((np.cos(theta), np.sin(theta)))

    for v in gcs.Vertices():
        if isinstance(v.set(), Point):
            ax.plot(v.set().x()[0], v.set().x()[1], "k", marker="o", markersize=5)
            ax.text(
                v.set().x()[0],
                v.set().x()[1] + 0.1,
                v.name(),
                horizontalalignment="center",
            )
        elif isinstance(v.set(), VPolytope):
            ax.fill(
                v.set().vertices()[0, :].T,
                v.set().vertices()[1, :].T,
                "lightgrey",
                edgecolor="k",
            )
            ax.text(
                np.mean(v.set().vertices()[0, :]),
                np.mean(v.set().vertices()[1, :]),
                v.name(),
                horizontalalignment="center",
                verticalalignment="center",
            )
        elif isinstance(v.set(), Hyperellipsoid):
            aball = AffineBall(v.set())
            vertices = aball.B() @ circle_points + aball.center().reshape((2, 1))
            ax.fill(vertices[0, :].T, vertices[1, :].T, "lightgrey", edgecolor="k")
            ax.text(
                v.set().center()[0],
                v.set().center()[1],
                v.name(),
                horizontalalignment="center",
                verticalalignment="center",
            )

    for e in gcs.Edges():
        # Solve a small program to draw the edges
        prog = MathematicalProgram()
        prog.AddDecisionVariables(e.xu())
        e.u().set().AddPointInSetConstraints(prog, e.xu())
        prog.AddDecisionVariables(e.xv())
        e.v().set().AddPointInSetConstraints(prog, e.xv())
        cost = prog.NewContinuousVariables(1, "cost")[0]
        prog.AddLorentzConeConstraint(np.concatenate(([cost], e.xu() - e.xv())))
        prog.AddLinearCost(cost)

        result = Solve(prog)
        assert result.is_success()
        ax.add_patch(
            FancyArrowPatch(
                result.GetSolution(e.xu()),
                result.GetSolution(e.xv()),
                arrowstyle="->",
                mutation_scale=20,
                color="k",
            )
        )


def toy_example():
    gcs = GraphOfConvexSets()

    source = gcs.AddVertex(Point([0, 0]), "source")
    vertices = np.array([[1, 1, 3, 3], [0, -2, -2, -1]])
    p1 = gcs.AddVertex(VPolytope(vertices), "p1")
    vertices = np.array([[4, 5, 3, 2], [-2, -4, -4, -3]])
    p2 = gcs.AddVertex(VPolytope(vertices), "p2")
    vertices = np.array([[2, 1, 2, 4, 4], [2, 3, 4, 4, 3]])
    p3 = gcs.AddVertex(VPolytope(vertices), "p3")
    e1 = gcs.AddVertex(Hyperellipsoid(np.eye(2), [4, 1]), "e1")
    e2 = gcs.AddVertex(Hyperellipsoid(np.diag([0.5, 1]), [7, -2]), "e2")
    vertices = np.array([[5, 7, 6], [4, 4, 3]])
    p4 = gcs.AddVertex(VPolytope(vertices), "p4")
    vertices = np.array([[7, 8, 9, 8], [2, 2, 3, 4]])
    p5 = gcs.AddVertex(VPolytope(vertices), "p5")
    target = gcs.AddVertex(Point([9, 0]), "target")

    gcs.AddEdge(source, p1)
    gcs.AddEdge(source, p2)
    gcs.AddEdge(source, p3)
    gcs.AddEdge(p1, e2)
    gcs.AddEdge(p2, p3)
    gcs.AddEdge(p2, e1)
    gcs.AddEdge(p2, e2)
    gcs.AddEdge(p3, p2)  # removing this changes the asymptotic behavior.
    gcs.AddEdge(p3, e1)
    gcs.AddEdge(p3, p4)
    gcs.AddEdge(e1, e2)
    gcs.AddEdge(e1, p4)
    gcs.AddEdge(e1, p5)
    gcs.AddEdge(e2, e1)
    gcs.AddEdge(e2, p5)
    gcs.AddEdge(e2, target)
    gcs.AddEdge(p4, p3)
    gcs.AddEdge(p4, e2)
    gcs.AddEdge(p4, p5)
    gcs.AddEdge(p4, target)
    gcs.AddEdge(p5, e1)
    gcs.AddEdge(p5, target)

    # |xu - xv|₂²
    for e in gcs.Edges():
        diff = e.xu() - e.xv()
        e.AddCost(diff.dot(diff))

    # First solve the convex relaxation
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.preprocessing = False
    result = gcs.SolveShortestPath(source, target, options)
    assert result.is_success()
    print("")
    print(
        f"Solution lower bound (from the relaxation): \t\t{result.get_optimal_cost()}"
    )

    # Now solve it again, with rounding enabled (to find a feasible solution)
    options.max_rounded_paths = 5
    result = gcs.SolveShortestPath(source, target, options)
    assert result.is_success()
    print(
        f"Solution upper bound (from a feasible solution): \t{result.get_optimal_cost()}"
    )
    # If the lower bound and upper bound are equal, then the solution obtained from the
    # relaxation is optimal.

    # Solve without convex_relaxation
    options.convex_relaxation = False
    result = gcs.SolveShortestPath(source, target, options)

    fig, ax = plt.subplots()
    Plot2dGraphOfConvexSets(gcs, ax)

    path = gcs.GetSolutionPath(source, target, result)
    print("Shortest path: ", end="")
    for e in path:
        vertices = np.vstack((result.GetSolution(e.xu()), result.GetSolution(e.xv())))
        ax.plot(
            vertices[:, 0],
            vertices[:, 1],
            "darkred",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=5,
        )

    ax.set_aspect("equal")
    ax.axis("off")


toy_example()

plt.show()
