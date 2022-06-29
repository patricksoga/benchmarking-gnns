from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import dgl
import networkx as nx
import seaborn as sb
from collections import Counter

def plot_similarity(g, EigVecs, EigVals):
    ng = dgl.to_networkx(g)
    # ng = nx.cycle_graph(10)
    # ng = nx.grid_graph((20, 20))
    pprint(ng.edges)
    pprint(nx.is_directed(ng))
    plt.figure("Graph")
    nx.draw(ng)

    # L = nx.laplacian_matrix(ng)
    # EigVals, EigVecs = np.linalg.eig(L.toarray())
    # idx = EigVals.argsort() # increasing order
    # EigVals, EigVecs = EigVals[idx], np.real(EigVecs[:,idx])

    similarities = np.transpose(EigVecs) @ EigVecs

    plt.figure("Similarities")
    ax = sb.heatmap(similarities, cmap="PiYG")
    ax.invert_yaxis()

    eigvec_1 = EigVecs[:, -2]
    eigvec_2 = EigVecs[:, -1]

    edges_x = []
    edges_y = []
    for e in ng.edges:
        source, sink = e[0], e[1]
        source_x = eigvec_1[source]
        source_y = eigvec_2[source]

        sink_x = eigvec_1[sink]
        sink_y = eigvec_2[sink]

        edges_x.append((source_x, sink_x))
        edges_y.append((source_y, sink_y))

    pprint(Counter(EigVals))
    plt.figure("Embedding with 2nd and 3rd ")
    sb.scatterplot(eigvec_1, eigvec_2)
    for e in zip(edges_x, edges_y):
        x, y = e
        plt.plot(x, y, c="red", mfc="blue", markersize=20, linewidth=0.5)

    plt.xlabel("2nd smallest eigenvector")
    plt.ylabel("3rd smallest eigenvector")
    plt.gca().set_aspect('equal')

    plt.show()
