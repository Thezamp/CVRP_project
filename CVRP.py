import osmnx as ox
import networkx as nx
import pandas as pd

from converters import convert_to_CVRP, convert_to_CARP
from solver import solve_CVRP, save_route

pd.options.mode.chained_assignment = None  # default='warn'

### Stats ###
all_clients = 555000
population = 8036

### production pro capita
production_py = 86.2  # kg of bio/ab, 2018
production_pw = production_py / 52  # production per week
production = production_pw / 3  # per abitant per collection

# vehicles
total_trucks = 280
trucks = total_trucks * population // all_clients  # 15, but not enough if no emptying is considered
capacity = 955  # nissan cabstar 35.11
trucks = round(population * production / capacity)

# depot
depot_node = 267

# speed limits assumptions
speed_limits_dict = {'living_street': 20, 'primary': 70, 'primary_link': 70, 'residential': 30, 'road': 30,
                     'secondary': 50, 'secondary_link': 50, 'tertiary': 50, 'tertiary_link': 50, 'unclassified': 50}


def clean_graph_from_osm(query):
    # load the graph
    g = ox.graph_from_place(query, network_type='drive', simplify=True)

    # rename nodes
    g = nx.convert_node_labels_to_integers(g)

    # As there are some errors in osm, keep maximally connected graph
    Gcc = sorted(nx.strongly_connected_components(g), key=len, reverse=True)
    G0 = g.subgraph(Gcc[0])

    # set the depot node as node 0
    remap_dict = {0: depot_node, depot_node: 0}
    G0 = nx.relabel_nodes(G0, remap_dict)

    # general fixes due to the format
    df = nx.to_pandas_edgelist(G0)
    # remove lists
    df['highway'] = df.highway.apply(lambda x: x[0] if isinstance(x, list) else x)
    df['maxspeed'] = df.maxspeed.apply(lambda x: x[0] if isinstance(x, list) else x)

    # deal with missing speed limits

    df['maxspeed'] = df['maxspeed'].fillna(df['highway'].map(speed_limits_dict))

    #return df[['source', 'target', 'maxspeed', 'length', 'highway', 'name']]
    return df,G0

def get_coeff(edgelist):
    mandatory = edgelist.loc[
        edgelist.highway.isin(['residential', 'tertiary', 'living_street', 'road', 'unclassified'])]

    # divide the arcs in 2 types, inhabited and movement, calculate total length
    movement_type_length = mandatory.loc[mandatory.highway.isin(['tertiary', 'road', 'unclassified'])].length.sum()
    inhabited_type_length = mandatory.loc[mandatory.highway.isin(['living_street', 'residential'])].length.sum()

    # hypothesis on distribution
    inhabited_type_pop = 0.8 * population
    movement_type_pop = 0.2 * population

    abitation_coeff_inhabited = inhabited_type_pop / inhabited_type_length
    abitation_coeff_movement = movement_type_pop / movement_type_length
    return abitation_coeff_inhabited, abitation_coeff_movement


def map_population(l, t, ci, cm):
    if t in ['tertiary', 'road', 'unclassified']:
        return l * cm
    elif t in ['living_street', 'residential']:
        return l * ci
    else:
        return 0



def main(query, extract_graph=False):
    edgelist, G_map = clean_graph_from_osm(query)
    if extract_graph:

        c_inh, c_move = get_coeff(edgelist)
        edgelist['population'] = edgelist.apply(lambda x: map_population(x.length, x.highway, c_inh,
                                                                         c_move), axis=1)
        edgelist['request'] = edgelist['population'] * production

        # cost in time: time to traverse the road + time for each gathering (hyp: 20 second "per person")
        edgelist['cost'] = edgelist.length.astype(float) / (edgelist.maxspeed.astype(float) * 1000) * 60 + \
                           edgelist.population * 1 / 3
        edgelist = edgelist.drop_duplicates(subset=['source', 'target'])
        # CARP info
        R_df = edgelist.loc[edgelist.highway.isin(['unclassified', 'residential', 'tertiary', 'living_street', 'road'])]
        R_df['node'] = R_df.apply(lambda x: f'{(x.source, x.target)}', axis=1)
        G_CARP = nx.from_pandas_edgelist(edgelist, edge_attr=['request', 'cost'])
        # R_CARP = nx.from_pandas_edgelist(R_df, edge_attr=['request', 'cost'])

        # ACVRP
        edgelist.to_csv('graphs_data/full_graph.csv')
        G = convert_to_CVRP(G_CARP, R_df)
        edges_vrp = nx.to_pandas_edgelist(G)
        edges_vrp.to_csv('graphs_data/Graph_CVRP.csv')

        R_df.to_csv('graphs_data/mandatory_edges.csv')
    else:
        edges_vrp = pd.read_csv('graphs_data/Graph_CVRP.csv')

        R_df = pd.read_csv('graphs_data/mandatory_edges.csv')

    G = nx.from_pandas_edgelist(edges_vrp[['source', 'target', 'cost']], edge_attr=['cost'])
    R_df = R_df[['node', 'request']]

    routes = solve_CVRP(K = trucks, G=G, R_df=R_df, capacity=capacity)
    colors = ['r', 'g', 'b', 'y', 'm']
    for r in range(5):
        save_route(G_map,routes[r],colors[r],r)
    return



if __name__ == '__main__':
    main('Caerano, Italy', extract_graph=False)
