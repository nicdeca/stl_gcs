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


def classic_shortest_path():
    gcs = GraphOfConvexSets()

    v = []
    for i in range(5):
        # The value of the point doesn't matter here
        v.append(gcs.AddVertex(Point([0.0]), f"v{i}"))

    v0_to_v1 = gcs.AddEdge(v[0], v[1])
    v0_to_v1.AddCost(3.0)
    gcs.AddEdge(v[1], v[0]).AddCost(1.0)
    gcs.AddEdge(v[0], v[2]).AddCost(4.0)
    gcs.AddEdge(v[1], v[2]).AddCost(1.0)
    v0_to_v3 = gcs.AddEdge(v[0], v[3])
    v0_to_v3.AddCost(1.0)
    v3_to_v2 = gcs.AddEdge(v[3], v[2])
    v3_to_v2.AddCost(1.0)
    gcs.AddEdge(v[1], v[4]).AddCost(2.5)  # Updated from original to break symmetry.
    v2_to_v4 = gcs.AddEdge(v[2], v[4])
    v2_to_v4.AddCost(3.0)
    gcs.AddEdge(v[0], v[4]).AddCost(6.0)

    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.preprocessing = False

    result = gcs.SolveShortestPath(v[0], v[4], options)
    assert result.is_success()

    path = gcs.GetSolutionPath(v[0], v[4], result)
    print("Shortest path:")
    for e in path:
        print(f"{e.u().name()} --> ", end="")
    print(path[-1].v().name())
    print(f"Path length = {result.get_optimal_cost()}")

    # ---- Step 7. Graphviz visualization options (if you want to dump to .dot) ----
    graphviz_options = GcsGraphvizOptions()
    graphviz_options.show_slacks = True
    graphviz_options.show_vars = True
    graphviz_options.show_flows = True
    graphviz_options.show_costs = True
    graphviz_options.scientific = False
    graphviz_options.precision = 3

    # The command below creates a Graphviz representation of the GCS, which can then be saved to a file, as done below, and then rendered to pdf as "dot -Tpdf gcs.dot -o gcs.pdf"
    graph_str = gcs.GetGraphvizString(result, graphviz_options, path)

    with open("gcs.dot", "w") as f:
        f.write(graph_str)

    # Render directly in Python
    graph = graphviz.Source(graph_str)
    graph.render("gcs_graph", format="png", cleanup=True)  # saves to gcs_graph.png
    graph.view("gcs_graph")  # opens with your system’s viewer


classic_shortest_path()
