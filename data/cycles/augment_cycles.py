from cgi import print_arguments
import pickle
import dgl
from matplotlib import pyplot as plt
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

def get_avg_pos_neg(graphs, labels):
    pos_graphs = [graph for graph, label in zip(graphs, labels) if label.item() == 1]
    positive_len = len(pos_graphs)
    avg_pos = sum([graph.number_of_nodes() for graph in pos_graphs])/positive_len

    neg_graphs = [graph for graph, label in zip(graphs, labels) if label.item() == 0]
    neg_len = len(neg_graphs)
    avg_neg = sum([graph.number_of_nodes() for graph in neg_graphs])/neg_len

    return avg_pos, avg_neg

def get_avg_cycle_basis_len(graphs, labels):
    pos_graphs_nx = [id_to_str(dgl.to_networkx(graph)) for graph, label in zip(graphs, labels) if label.item() == 1]

    # network = Network('1000px', '1000px')
    # random_idx = random.randint(0, len(pos_graphs_nx))
    # visualize(pos_graphs_nx[random_idx], network)
    # exit()

    pos_graph_bases = list(map(nx.cycle_basis, pos_graphs_nx))
    pos_graph_bases_lens = [len(min(basis, key=len)) for basis in pos_graph_bases]
    pos_graph_bases_lens_ = [len(max(basis, key=len)) for basis in pos_graph_bases]
    # pos_graph_bases_lens = list(map(len, pos_graph_bases))

    # neg_graph_bases_lens = list(map(len, neg_graph_bases))

    # with open('./peer.txt', 'w+') as f:
    #     f.write(pformat(sorted(pos_graph_bases_lens)))
    # exit()
    # import matplotlib.pyplot as plt
    # plt.hist(pos_graph_bases_lens, bins=15)
    # plt.show()

    avg_cycle_basis_len_pos = sum(pos_graph_bases_lens)/len(pos_graph_bases)
    avg_cycle_basis_len_pos_ = sum(pos_graph_bases_lens_)/len(pos_graph_bases)

    return avg_cycle_basis_len_pos, avg_cycle_basis_len_pos_

def get_avg_diameters(graphs, labels):
    pos_graphs_nx = [id_to_str(dgl.to_networkx(graph)) for graph, label in zip(graphs, labels) if label.item() == 1]
    pos_graphs_nx = [graph.subgraph(max(nx.connected_components(graph), key=len).copy()) for graph in pos_graphs_nx]
    pos_graph_diameters = list(map(nx.diameter, pos_graphs_nx))
    return sum(pos_graph_diameters)/len(pos_graph_diameters)

def add_cycles(graph: nx.Graph, cycle_range=[21, 30]):
    cycle_basis = nx.cycle_basis(graph)
    nodes = list(map(int, graph.nodes()))
    max_node_id = max(nodes)
    m = random.choice(np.arange(cycle_range[0], cycle_range[1]))

    # print('m: ', m)
    # visualize(graph, Network(height="1000px", width="1000px"))
    # input()
    if m != 0:
        for basis in cycle_basis:
            # edge_1 = (basis[-2], max_node_id+1)
            # edge_2 = (max_node_id+1, basis[-1])

            # graph.remove_edges_from([(basis[-2], basis[-1])])
            # graph.add_edges_from([edge_1, edge_2])

            # print('basis len before: ', len(basis))
            nodes_to_add = np.arange(max_node_id + 1, max_node_id + 1 + m)
            nodes_to_add = list(map(str, nodes_to_add))

            edge_set = []

            for i, node in enumerate(nodes_to_add[:-1] if m > 1 else nodes_to_add):
                if m > 1:
                    edge_set.append((node, nodes_to_add[i+1]))
                else:
                    edge_set.append((node, nodes_to_add[0]))

            idx = len(basis)//2
            edge_set.append((basis[idx], edge_set[0][0]))
            edge_set.append((edge_set[-2][1], basis[idx+1]))
            graph.remove_edges_from([(basis[idx], basis[idx+1])])
            graph.add_edges_from(edge_set)


            max_node_id = max_node_id + 1 + m
        
    # cycle_basis = nx.cycle_basis(graph)
    # for basis in cycle_basis:
    #     print('basis len after: ', len(basis))
    # print()
    # visualize(graph, Network(height="1000px", width="1000px"))
    # input()

        # cycle = [basis[0], *nodes_to_add, basis[-1]]
        # nx.add_cycle(graph, cycle)

        # extend one of the nodes in the new cycle with a path.
        # for node in nodes_to_add:
        #     should_extend = random.randint(0, 10)
        #     if should_extend != 0:
        #         continue

        #     path = [node]
        #     for _ in range(random.choice(cycle_range)):
        #         max_node_id += 1
        #         path.append(max_node_id)
            
        #     nx.add_path(graph, path)

        # max_node_id += 1
    
    # cycle_bases = nx.cycle_basis(graph)
    # good_bases = [set(basis) for basis in cycle_bases if len(basis) >= m]
    # bad_bases = [set(basis) for basis in cycle_bases if len(basis) != m]

    # for basis in bad_bases:
    #     intersections = [basis.intersection(good_basis) for good_basis in good_bases]
    #     for node in basis:
    #         if all(node not in intersection for intersection in intersections):
    #             try:
    #                 graph.remove_node(str(node))
    #             except:
    #                 pass

    new_graph = nx.Graph()
    new_graph.add_edges_from(graph.edges())

    return new_graph

def add_nodes(graph: nx.Graph, value_range=[26, 30], at_end=False):
    largest_cc = max(nx.connected_components(graph), key=len)
    g = graph.subgraph(largest_cc).copy()

    rand_root = random.choice(list(g.nodes()))

    if at_end:
        for node in g.nodes():
            if len(graph[node]) != 1:
                continue
            rand_root = node
            break

    max_node = max(list(map(int, graph.nodes())))
    path = [rand_root]
    for _ in range(random.choice(value_range)):
        max_node += 1
        path.append(max_node)

    try:
        nx.add_path(graph, path)
    except:
        nx.draw(g)
        plt.show()

    # for node in path:
    #     should_extend = random.randint(0, 1)
    #     if should_extend == 0:
    #         continue
        
    #     # extend_n = random.randint(1, 2)
    #     path = [node]
    #     # for _ in range(extend_n):
    #     for _ in range(random.choice(value_range)):
    #         max_node += 1
    #         path.append(max_node)

    #     nx.add_path(graph, path)
    return graph

def add_dgl(g: nx.Graph):
    edges = g.edges()
    srcs = torch.tensor([u for u, _ in edges])
    dsts = torch.tensor([v for _, v in edges])

    graph = dgl.DGLHeteroGraph()
    graph.add_edges(srcs, dsts)

    graph.ndata['feat'] = torch.ones(graph.number_of_nodes(), 1, dtype=torch.float)
    graph.edata['feat'] = torch.ones(graph.number_of_edges(), 1, dtype=torch.float)

    return graph

def show_stats(dataset):
    avg_pos_nodes, avg_neg_nodes = get_avg_pos_neg(dataset.graph_lists, dataset.graph_labels)
    print('Average |V| (cyclic): ', avg_pos_nodes)
    print('Average |V| (acyclic): ', avg_neg_nodes)
    a = get_avg_cycle_basis_len(dataset.graph_lists, dataset.graph_labels)
    print(f'Average cycle basis length: {a[0]}, {a[1]}')
    print('Average diameter: ', get_avg_diameters(dataset.graph_lists, dataset.graph_labels))

def generate_plain(g: nx.Graph, n=100, cycle=True):
    if cycle:
        cycle_graph = nx.cycle_graph(20)
        remaining_nodes = n-20
        g = cycle_graph
        edges = []
        count = 9
        for i in range(remaining_nodes):
            edges.append((i+count, i+count + 1))

        g.add_edges_from(edges)
    else:
        path_graph = nx.path_graph(100)
        g = path_graph
    return g

def main():
    train, val, test = pickle.load(open('/Users/psoga/Documents/projects/benchmarking-gnns/data/cycles/CYCLES_6_56.pkl', 'rb'))
    network = Network('1000px', '1000px')

    low, high = (0, 8)
    # low, high = (6, 10)
    # show_stats(val)
    # exit()

    for i, dataset in enumerate([train, val, test]):
        new_graph_lists = []
        for j, tup in tqdm.tqdm(enumerate(dataset)):
            graph, label, _ = tup

            if label.item() == 0:
                g = dgl.to_networkx(graph)
                g = nx.to_undirected(g)
                g = id_to_str(g)
                # g = generate_plain(g, n=100, cycle=False)
                g = add_nodes(g, [int(low*4.6), int(high*4.6)], at_end=True)
                # g = add_nodes(g, [int(low*3.2), int(high*3.2)])
                # g = add_nodes(g, [int(low*2.2), int(high*2.2)], at_end=True)
                # g = add_nodes(g, [low+5, high+5])
                g = str_to_id(g)
                g = g.to_directed()
                graph = add_dgl(g)
                new_graph_lists.append(graph)
                continue

            g = dgl.to_networkx(graph)
            g = nx.to_undirected(g)
            g = id_to_str(g)
            # g = generate_plain(g, n=100, cycle=True)
            g = add_cycles(g, [low, high])
            g = add_nodes(g, [int(low*2.4), int(high*2.4)], at_end=True)
            # g = add_nodes(g, [int(low*2.2), int(high*2.2)])
            # g = add_nodes(g, [int(low*2.2), int(high*2.2)], at_end=True)
            g = str_to_id(g)
            g = g.to_directed()
            # if j > 100:
            #     visualize(g, network)
            #     exit()

            graph = add_dgl(g)
            new_graph_lists.append(graph)

        if i == 0:
            train.graph_lists = new_graph_lists
            # random_index = random.randint(0, len(train.graph_lists))
            # visualize(dgl.to_networkx(train.graph_lists[random_index]), network)
            # exit()
        elif i == 1:
            val.graph_lists = new_graph_lists
            # show_stats(val)
        elif i == 2:
            test.graph_lists = new_graph_lists

    datasets = (train, val, test)

    print('Stats for train, val, test: ')

    for d in datasets:
        show_stats(d)
        print()

    pickle.dump(datasets, open('/Users/psoga/Documents/projects/benchmarking-gnns/data/cycles/CYCLES_-2_56.pkl', 'wb'))


if __name__ == "__main__":
    main()