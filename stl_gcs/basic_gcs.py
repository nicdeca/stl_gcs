import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import graphviz

from pydrake.all import (
    Hyperellipsoid,
    VPolytope,
    Point,
    AffineBall,
    MathematicalProgram,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
    GcsGraphvizOptions,
    Solve,
)


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


# Utility: plot a VPolytope
def plot_vpoly(ax, vpoly, name=None, color="lightblue"):
    verts = vpoly.vertices()
    ax.fill(verts[0, :], verts[1, :], facecolor=color, edgecolor="k", alpha=0.5)
    if name is not None:
        cx, cy = np.mean(verts, axis=1)
        ax.text(cx, cy, name, ha="center", va="center")


# ---- Step 1. Define polytopic sets (boxes in 2D) ----
# 1) Build polytopic "rooms" + corridor
Sv = np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]])
S = VPolytope(Sv)

R1v = np.array([[1.5, 1.5, 2.5, 2.5], [0.0, 1.0, 1.0, 0.0]])
R1 = VPolytope(R1v)

R2v = np.array([[3.0, 3.0, 4.0, 4.0], [0.0, 1.0, 1.0, 0.0]])
R2 = VPolytope(R2v)

Cv = np.array([[4.5, 4.5, 5.5, 5.5], [0.0, 1.0, 1.0, 0.0]])
C = VPolytope(Cv)

R3v = np.array([[6.0, 6.0, 7.0, 7.0], [0.0, 1.0, 1.0, 0.0]])
R3 = VPolytope(R3v)

Tv = np.array([[7.5, 7.5, 8.5, 8.5], [0.0, 1.0, 1.0, 0.0]])
T = VPolytope(Tv)

# # --- Plot everything ---
# fig, ax = plt.subplots()
# plot_vpoly(ax, S, "S", "lightgreen")
# plot_vpoly(ax, R1, "R1", "lightblue")
# plot_vpoly(ax, R2, "R2", "orange")
# plot_vpoly(ax, C, "C", "lightgrey")
# plot_vpoly(ax, R3, "R3", "violet")
# plot_vpoly(ax, T, "T", "pink")

# ax.set_aspect("equal")
# ax.set_xlim(-0.5, 9.5)
# ax.set_ylim(-0.5, 2.0)
# ax.axis("off")
# plt.show()


# ---- Step 2. Build GCS ----
gcs = GraphOfConvexSets()

# ---- Step 2.1 Add convex sets as vertices to the graph ----
vS = gcs.AddVertex(S, name="S")
vR1 = gcs.AddVertex(R1, name="R1")
vR2 = gcs.AddVertex(R2, name="R2")
vC = gcs.AddVertex(C, name="Corridor")
vR3 = gcs.AddVertex(R3, name="R3")
vT = gcs.AddVertex(T, name="T")

# ---- Step 2.2 Connect overlapping sets ----
# Add candidate edges (we connect sets that overlap or are reachable)
gcs.AddEdge(vS, vR1)
gcs.AddEdge(vR1, vR2)
gcs.AddEdge(vR2, vC)
gcs.AddEdge(vC, vR3)
gcs.AddEdge(vR3, vT)
gcs.AddEdge(vR1, vC)
gcs.AddEdge(vR2, vR3)


# ---- Step 3 Add costs for edges ----
# |xu - xv|₂²
for e in gcs.Edges():
    diff = e.xu() - e.xv()
    e.AddCost(diff.dot(diff))


# ---- Step 4. Set options ----
options = GraphOfConvexSetsOptions()
options.convex_relaxation = True
# max_rounding_trials not used if convex_relaxation is false
options.max_rounding_trials = 5  # Maximum number of distinct paths to compare during random rounding; only the lowest cost path is returned.
# If convex_relaxation is false or this is less than or equal to zero, rounding is not performed. If max_rounded_paths=nullopt, then each GCS method is free to choose an appropriate default.
options.preprocessing = True  # Performs a preprocessing step to remove edges that cannot lie on the path from source to target.

# ---- Step 5. Solve convex relaxation shortest path ----
result = gcs.SolveShortestPath(vS, vT, options)
assert result.is_success()
print(f"Solution lower bound (from the relaxation): \t\t{result.get_optimal_cost()}")

print("Solver success:", result.is_success())
print("\nEdges and flows (from relaxation):")
for e in gcs.Edges():
    # edge name may be empty; use u->v if so
    name = e.name() if e.name() else f"{e.u().name()}->{e.v().name()}"
    # e.flow() is the flow decision variable for the edge
    try:
        flow_val = e.GetSolutionCost(result)
    except Exception:
        flow_val = None
    print(f"  {name:12s} | flow = {flow_val}")


# ---- Step 6. Recover discrete path ----
path = gcs.GetSolutionPath(source=vS, target=vT, result=result)

for v in path:
    x_val = result.GetSolution(v.xu())
    print(f"{v.name()} -> {x_val}")


# ----  Step 7. (Optional) Refine the solution using convex restriction. Now the path variables are fixed, so the problem is convex. Nonconvex constraints may be added here.
ref_res = gcs.SolveConvexRestriction(path, options=options, initial_guess=result)
print("Refined cost:", ref_res.get_optimal_cost())

print("")
print(
    f"Solution lower bound (from the convex restriction): \t\t{ref_res.get_optimal_cost()}"
)

path = gcs.GetSolutionPath(source=vS, target=vT, result=result)

for v in path:
    x_val = ref_res.GetSolution(v.xu())
    print(f"{v.name()} -> {x_val}")


fig, ax = plt.subplots()
Plot2dGraphOfConvexSets(gcs, ax)

path = gcs.GetSolutionPath(vS, vT, result)
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

# ---- Step 7. Graphviz visualization options (if you want to dump to .dot) ----
graphviz_options = GcsGraphvizOptions()
graphviz_options.show_slacks = True
graphviz_options.show_vars = True
graphviz_options.show_flows = True
graphviz_options.show_costs = True
graphviz_options.scientific = False
graphviz_options.precision = 3


# The command below creates a Graphviz representation of the GCS, which can then be saved to a file, as done below, and then rendered to pdf as "dot -Tpdf gcs.dot -o gcs.pdf"
graph_str = gcs.GetGraphvizString(ref_res, graphviz_options, path)

with open("gcs.dot", "w") as f:
    f.write(graph_str)


# Render directly in Python
graph = graphviz.Source(graph_str)
graph.render("gcs_graph", format="png", cleanup=True)  # saves to gcs_graph.png
graph.view("gcs_graph")  # opens with your system’s viewer


plt.show()
