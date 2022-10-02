import pickle
import dgl
import networkx as nx
import numpy as np
import random
import tqdm
import torch
from pyvis.network import Network

def id_to_str(graph: nx.Graph):
    g = nx.Graph()
    g.add_edges_from([(str(edge[0]), str(edge[1])) for edge in graph.edges()])
    g.add_nodes_from([str(id) for id in graph.nodes()])
    return g

def str_to_id(graph: nx.Graph):
    g = nx.Graph()
    g.add_edges_from([(int(edge[0]), int(edge[1])) for edge in graph.edges()])
    g.add_nodes_from([int(id) for id in graph.nodes()])
    return g

def visualize(graph: nx.Graph, network: Network):
    graph = id_to_str(graph)
    network.from_nx(graph)
    network.show('nx.html')

def add_cycles(graph: nx.Graph, cycle_range=[1, 12]):
    cycle_basis = nx.cycle_basis(graph)
    nodes = list(map(int, graph.nodes()))
    max_node_id = max(nodes)
    m = random.choice(cycle_range)

    new_cycles = []
    for basis in cycle_basis:
        nodes_to_add = np.arange(max_node_id + 1, max_node_id + 1 + m)
        nodes_to_add = list(map(str, nodes_to_add))

        new_cycles.append([basis[0], *nodes_to_add, basis[-1]])
        max_node_id = max_node_id + 1 + m
    
    for new_cycle in new_cycles:
        nx.add_cycle(graph, new_cycle)

    return graph

def main():
    train, val, test = pickle.load(open('/Users/psoga/Documents/projects/benchmarking-gnns/data/cycles/CYCLES_6_56.pkl', 'rb'))
    network = Network('1000px', '1000px')

    # test_train = pickle.load(open('./test_train.pkl', 'rb'))

    # graph = test_train[0]
    # graph = dgl.to_networkx(graph)
    # graph = nx.to_undirected(graph)
    # visualize(graph, network)
    # print(nx.cycle_basis(graph))
    # visualize(graph, network)

    # train_graphs = [x for x in train if x[1].item() == 1]
    # val_graphs = [x for x in val if x[1].item() == 1]
    # test_graphs = [x for x in test if x[1].item() == 1]

    for i, dataset in enumerate([train, val, test]):
        new_graph_lists = []
        for tup in tqdm.tqdm(dataset):
            graph, label, _ = tup

            if label.item() == 0:
                new_graph_lists.append(graph)
                continue
            g = dgl.to_networkx(graph)
            # print('inintial nodes: ', g.number_of_nodes())
            # print('initial edges: ', g.number_of_edges())
            g = nx.to_undirected(g)
            # min_spannin_forest = nx.minimum_spanning_tree(g)
            # min_spannin_forest = id_to_str(min_spannin_forest)
            # orig_g = id_to_str(g)
            # network = Network('1000px', '1000px')
            # network.from_nx(orig_g)
            g = id_to_str(g)
            g = add_cycles(g)
            g = str_to_id(g)
            g = g.to_directed()
            visualize(g, network)
            # print('nodes with cycles: ', g.number_of_nodes())
            # print('edges with cycles: ', g.number_of_edges())

            edges = g.edges()
            srcs = torch.tensor([u for u, _ in edges])
            dsts = torch.tensor([v for _, v in edges])

            graph.add_edges(srcs, dsts)
            # print(graph.number_of_nodes())
            # print(graph.number_of_edges())
            # exit()

            # g = dgl.from_networkx(g)

            graph.ndata['feat'] = torch.ones(graph.number_of_nodes(), 1, dtype=torch.float)
            graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1, dtype=torch.float)
            # dataset.graph_lists[i] = g
            new_graph_lists.append(graph)

            # network.from_nx(g)
            # network.show('nx.html')
        if i == 0:
            train.graph_lists = new_graph_lists
        elif i == 1:
            val.graph_lists = new_graph_lists
        elif i == 2:
            test.graph_lists = new_graph_lists

    pickle.dump((train, val, test), open('/Users/psoga/Documents/projects/benchmarking-gnns/data/cycles/CYCLES_-1_56.pkl', 'wb'))


if __name__ == "__main__":
    main()