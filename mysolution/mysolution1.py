import random
import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

n = 100
p = 0.05

# 2) Generowanie graf Erdos–Renyi
g = ig.Graph.Erdos_Renyi(n=n, p=p, directed=False)
g.vs["name"] = [f"v{i}" for i in range(n)]

# 3) Podsumowanie przed wagami
print("PODSUMOWANIE (przed ustawieniem wag)")
print(g.summary())
is_weighted_before = ("weight" in g.edge_attributes()) and (len(g.es["weight"]) > 0)
print("Czy graf jest ważony?", is_weighted_before)

# 4) Wylistuj wszystkie wierzchołki i krawędzie
print("\nWierzchołki:")
print(g.vs["name"])  # wierzchołki
print("\nKrawędzie (indeksy):")
print([e.tuple for e in g.es])  # krawędzie (krotki indeksów wierzchołków)

# 5) Ustaw losowe wagi krawędzi
weights = list(np.random.uniform(0.01, 1.0, size=len(g.es)))
g.es["weight"] = weights

# 6) Podsumowanie po ustawieniu wag
print("\nPODSUMOWANIE (po ustawieniu wag)")
print(g.summary())
is_weighted_after = ("weight" in g.edge_attributes()) and (len(g.es["weight"])>0)
print("Czy graf jest ważony?", is_weighted_after)

# pokaż pierwsze 30 krawędzi z wagami
print("\nPrzykładowe krawędzie z wagami (pierwsze 30):")
for i, e in enumerate(g.es[:30]):
    u, v = e.tuple
    print(i, g.vs[u]["name"], g.vs[v]["name"], f"weight={e['weight']:.4f}")

# 7) Stopnie węzłów, tabela, histogram
degrees = g.degree()
deg_df = pd.DataFrame({
    "vertex_index": range(n),
    "vertex_name": g.vs["name"],
    "degree": degrees
})
print("\nStopnie wszystkich węzłów:")
print(deg_df.to_string(index=False))

# Histogram stopni
plt.figure(figsize=(8, 5))
plt.hist(degrees, bins=range(min(degrees), max(degrees)+2))
plt.xlabel("Stopień węzła")
plt.ylabel("Liczba węzłów")
plt.title(f"Histogram stopni (n={n}, p={p})")
plt.grid(True, linestyle='--', alpha=0.3)
plt.savefig("results/ex1_degree_histogram_igraph.png")
plt.close()
print("Histogram zapisano jako results/ex1_degree_histogram_igraph.png")

# 8) Komponenty spójne
components = g.components()
print("\nLiczba komponentów spójnych:", len(components))
print("Rozmiary komponentów:", components.sizes())

# 9) PageRank i wizualizacja (rozmiar węzłów ~ PageRank)
pagerank_scores = g.pagerank(weights="weight")
pr_min, pr_max = min(pagerank_scores), max(pagerank_scores)
if pr_max - pr_min > 0:
    vertex_sizes = [8 + 37 * ((pr - pr_min) / (pr_max - pr_min)) for pr in pagerank_scores]
else:
    vertex_sizes = [12]*len(pagerank_scores)

g.vs["size"] = vertex_sizes
initial_pos = np.random.default_rng(random_seed).random((g.vcount(), 2))
layout = g.layout_fruchterman_reingold(seed=initial_pos)


visual_style = {
    "vertex_size": g.vs["size"],
    "vertex_label": None,
    "edge_width": [max(0.2, 2.5*w) for w in g.es["weight"]],
    "layout": layout,
    "bbox": (1200, 900),
    "margin": 50
}
ig.plot(g, "results/ex1_graph_pagerank_igraph.png", **visual_style)
print("Wizualizacja zapisana jako results/ex1_graph_pagerank_igraph.png")
