import time
import networkx as nx


def convert_to_CARP(G, R):
    # mandatory_edges_list = R[['source','target']].to_records(index=False)
    it = 0
    V_r = set(R.source.tolist() + R.target.tolist())
    V_r.add(0)
    print(len(V_r))
    start = time.time()
    new_graph = {}
    for s in V_r:
        connections = {}
        for t in [target for target in V_r if target != s]:
            it += 1
            if (it % 10000 == 0):
                print(it)
            if G.has_edge(s, t):
                connections[t] = {'cost': G.get_edge_data(s, t)['cost']}
            else:
                connections[t] = {'cost': nx.shortest_path_length(G, s, t, weight='cost')}
        new_graph[s] = connections

    G_out = nx.from_dict_of_dicts(new_graph)
    print(time.time() - start)
    return G_out


def convert_to_CVRP(G, R):
    mandatory_edges_list = R[['source', 'target']].to_records(index=False)
    start = time.time()
    new_graph = {}
    # depot node
    connections = {}
    cnt = 1
    for (h, l) in mandatory_edges_list:
        connections[(h, l)] = {'cost': nx.shortest_path_length(G, 0, h, weight='cost')}

    new_graph[(0, 0)] = connections

    # edges map to nodes
    nodes_attr = {}
    for (i, j) in mandatory_edges_list:
        nodes_attr[f'({i}, {j})'] = R.loc[(R.source == i) & (R.target == j)].request.values[0]
        cnt += 1
        if (cnt % 100 == 0):
            print(cnt)
        connections = {}
        edge_cost = G.get_edge_data(i, j)['cost']
        for (h, l) in mandatory_edges_list:
            spl = nx.shortest_path_length(G, j, h, weight='cost')
            connections[(h, l)] = {'cost': spl + edge_cost}
        if j == 0:
            connections[(0, 0)] = {'cost': edge_cost}
        new_graph[(i, j)] = connections

    G_out = nx.from_dict_of_dicts(new_graph)
    # add the request info
    nx.set_node_attributes(G_out, nodes_attr, 'request')

    print(time.time() - start)
    return G_out
