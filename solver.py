import numpy as np

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt


def solve_CVRP(K, G, R_df, capacity):
    # update the graph nodes with the requests values
    request_for_node = dict(zip(R_df.node, R_df.request))
    request_for_node['(0, 0)'] = 0
    nx.set_node_attributes(G, request_for_node, 'request')

    #convert names to simple number, save the back_conversion
    G = nx.convert_node_labels_to_integers(G, label_attribute='matching')

    cost_matrix = nx.to_numpy_matrix(G, weight='cost')

    # request_dict with renamed nodes
    request_dict = dict(G.nodes(data='request'))
    request_values = np.array(list(request_dict.values()))
    conversion_dict = dict(G.nodes(data='matching'))

    # keep track of clients to service
    to_service = list(range(len(request_values)))
    to_service.remove(0)

    # initialize routes
    routes = {}
    capacities = capacity * np.ones(K)

    # find pivots
    for k in range(K):
        pivot = np.argmax(request_values)
        routes[k] = [0, pivot, 0]
        capacities[k] -= request_values[pivot]
        request_values[pivot] = -1
        to_service.remove(pivot)

    # heuristically add the clients

    while len(to_service) > 0:
        if len(to_service) % 50 == 0:
            print(len(to_service))
        best_k = -1
        best_pos = -1
        best_c = -1
        best_extramil = float('inf')

        #find what (client, route, position) provides the minimal increase in mileage

        # for each client
        for c in to_service:
            # for each route
            for k in range(K):
                # if the capacity is respected
                if capacities[k] - request_values[c] > 0:
                    # for each position
                    for i in range(1, len(routes[k]) - 1):

                        #extramileage = cost a->c + cost c->b - cost a->b

                        extramileage = cost_matrix[routes[k][i - 1], c] + cost_matrix[c, routes[k][i]] - cost_matrix[
                            routes[k][i - 1], routes[k][i]]
                        if extramileage < best_extramil:
                            best_extramil = extramileage
                            best_k = k
                            best_pos = i
                            best_c = c
        try:
            routes[best_k].insert(best_pos, best_c)
            to_service.remove(best_c)
            capacities[best_k] -= request_values[best_c]
        except KeyError:
            #the trucks were miscalculated
            return {},-1

    new_routes = {}
    for k in range(K):
        route = routes[k]
        new_routes[k] = [conversion_dict[v] for v in route]

    return new_routes,capacities


def save_route(G, route, c,n):
    path=[0]
    for i in range(1, len(route)):
        n0, n1 = eval(route[i - 1])
        n2, n3 = eval(route[i])
        # path.extend(ox.distance.shortest_path(G_full,n1,n2, weight='cost')) #0 ->135
        path.extend(ox.distance.shortest_path(G, n1, n2))
    path = path[1:]
    fig,ax = ox.plot.plot_graph_route(G, path, c)
    fig.savefig(f'results/route_{n}')
