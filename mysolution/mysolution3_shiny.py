import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from igraph import Graph
from shiny import App, ui, reactive, render

app_ui = ui.page_fluid(
    ui.h2("Symulacja rozprzestrzeniania informacji"),
    ui.input_slider("wij_factor", "Mnożnik prawdopodobieństwa aktywacji (10%-200%)", 10, 200, 100),
    ui.input_slider("max_iter", "Maksymalna liczba iteracji", 1, 50, 10),
    ui.output_plot("diffusion_plot")
)

df = pd.read_csv(
    "input/out.radoslaw_email_email",
    sep=r"\s+",
    header=None,
    skiprows=2,
    engine="python"
)
df = df.iloc[:, :2]
df.columns = ["source", "target"]
df["source"] = df["source"].astype(str)
df["target"] = df["target"].astype(str)

edges = list(df.itertuples(index=False, name=None))
g = Graph.TupleList(edges, directed=True, vertex_name_attr="name")
g.simplify(multiple=True, loops=True, combine_edges=None)

cnt_ij = df.groupby(["source", "target"]).size().reset_index(name="cnt_ij")
cnt_i = df.groupby("source").size().reset_index(name="cnt_i")
cnt_ij = cnt_ij.merge(cnt_i, on="source")
cnt_ij["weight"] = cnt_ij["cnt_ij"] / cnt_ij["cnt_i"]

weight_dict = {
    (str(row.source), str(row.target)):
    row.weight for row in cnt_ij.itertuples()
}
weights = []
for e in g.es:
    s = g.vs[e.source]["name"]  # string
    t = g.vs[e.target]["name"]  # string
    weights.append(weight_dict.get((s, t), 0.0))

g.es["weight"] = weights

N = len(g.vs)
num_initial = max(1, int(0.05 * N))


def independent_cascade_simulation(g, initial_set, max_iter=10, wij_factor=100):
    g.vs["activated"] = [False] * N
    for v in initial_set:
        g.vs[v]["activated"] = True

    activated_counts = [len(initial_set)]
    current_front = set(initial_set)
    factor = wij_factor / 100.0

    iter_count = 0
    while len(current_front) > 0 and iter_count < max_iter:
        next_front = set()
        attempted = set()

        for v in current_front:
            out_edges = g.es.select(_source=v)
            for e in out_edges:
                u = e.target
                if g.vs[u]["activated"]:
                    continue
                if u in attempted:
                    continue

                attempted.add(u)
                prob = min(1.0, e["weight"] * factor)
                if random.random() < prob:
                    g.vs[u]["activated"] = True
                    next_front.add(u)

        current_front = next_front
        activated_counts.append(sum(g.vs["activated"]))
        iter_count += 1

    return activated_counts


def server(input, output, session):

    @output
    @render.plot
    def diffusion_plot():
        # losowe 5% węzłów startowych
        initial_set = random.sample(range(N), num_initial)
        curve = independent_cascade_simulation(
            g,
            initial_set,
            max_iter=input.max_iter(),
            wij_factor=input.wij_factor()
        )

        plt.figure(figsize=(10,6))
        plt.plot(range(len(curve)), curve, color="blue", linewidth=2)
        plt.xlabel("Numer iteracji")
        plt.ylabel("Liczba aktywowanych węzłów")
        plt.title("Dyfuzja informacji w grafie")
        plt.grid(True)
        plt.tight_layout()
        return plt.gcf()


app = App(app_ui, server)
