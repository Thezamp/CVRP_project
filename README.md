# CDMO project
## Routing garbage collection in a small town

### Usage
See notebook for example.
Import prepare_graph, solve_CVRP and save_route

Define population, production, capacity and number of trucks.

First call prepare_graph to clean the results from osmnx query

Then solve_CVRP will return the routes and the final remaining capacities for the trucks

Save_route will save a graphical representation of the found routes