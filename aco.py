# import networkx as nx
# import csv
# import random
# import math

# # --------------------------
# # Step 1: Load CSV into a Graph
# # --------------------------
# G = nx.Graph()
# with open('network_edges.csv', 'r') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         src = int(row['source'])
#         dst = int(row['target'])
#         weight = float(row['weight']) 
#         G.add_edge(src, dst, weight=weight)

# # --------------------------
# # Step 2: ACO Parameters & Initialization
# # --------------------------
# num_ants = 20           # Number of ants per iteration
# num_iterations = 100    # Total iterations for the algorithm
# alpha = 1.0             # Influence of pheromone on decision making
# beta = 2.0              # Influence of heuristic (1/weight) on decision making
# evaporation_rate = 0.5  # Rate at which pheromones evaporate
# Q = 100                 # Constant for pheromone deposit calculation

# # Initialize pheromones: start with 1 on every edge (bidirectional for undirected graph)
# pheromones = {}
# for edge in G.edges():
#     pheromones[edge] = 1.0
#     pheromones[(edge[1], edge[0])] = 1.0

# # Define source and destination nodes
# source = 0
# destination = max(G.nodes())  # For example, choose the highest numbered node

# best_path = None
# best_cost = float('inf')

# # --------------------------
# # Step 3: Define Helper Function for Next Node Selection
# # --------------------------
# def select_next_node(current_node, visited):
#     # List all neighbor nodes not already visited
#     neighbors = [n for n in G.neighbors(current_node) if n not in visited]
#     if not neighbors:
#         return None
#     # Calculate probability for each neighbor based on pheromone and heuristic information
#     prob_list = []
#     total_prob = 0.0
#     for neighbor in neighbors:
#         edge = (current_node, neighbor)
#         tau = pheromones[edge]  # Pheromone level
#         weight = G[current_node][neighbor]['weight']
#         heuristic = 1.0 / weight  # Lower cost means higher attractiveness
#         prob = (tau ** alpha) * (heuristic ** beta)
#         prob_list.append((neighbor, prob))
#         total_prob += prob

#     # Randomly select next node weighted by the calculated probabilities
#     r = random.random() * total_prob
#     cumulative = 0.0
#     for neighbor, prob in prob_list:
#         cumulative += prob
#         if cumulative >= r:
#             return neighbor
#     return neighbors[-1]  # Fallback in case of rounding issues

# # --------------------------
# # Step 4: Main ACO Loop
# # --------------------------
# for iteration in range(num_iterations):
#     all_paths = []
#     all_costs = []
#     for ant in range(num_ants):
#         path = [source]
#         current_node = source
        
#         # Build a path until destination is reached (or break if stuck)
#         while current_node != destination:
#             next_node = select_next_node(current_node, path)
#             if next_node is None:
#                 path = None  # Dead end encountered
#                 break
#             path.append(next_node)
#             current_node = next_node
#             # Avoid potential infinite loops by limiting path length
#             if len(path) > len(G.nodes()):
#                 path = None
#                 break
        
#         # If a valid path is found, compute its total cost
#         if path is not None:
#             cost = 0
#             for i in range(len(path) - 1):
#                 cost += G[path[i]][path[i+1]]['weight']
#             all_paths.append(path)
#             all_costs.append(cost)
#             if cost < best_cost:
#                 best_cost = cost
#                 best_path = path

#     # --------------------------
#     # Pheromone Evaporation: Reduce pheromone levels on all edges
#     # --------------------------
#     for edge in pheromones:
#         pheromones[edge] *= (1 - evaporation_rate)
    
#     # --------------------------
#     # Pheromone Update: Increase pheromone on the paths found by ants
#     # --------------------------
#     for path, cost in zip(all_paths, all_costs):
#         pheromone_deposit = Q / cost  # Better (lower cost) paths get more pheromone
#         for i in range(len(path) - 1):
#             edge = (path[i], path[i+1])
#             # Update both directions for undirected graph
#             pheromones[edge] += pheromone_deposit
#             pheromones[(edge[1], edge[0])] += pheromone_deposit

# # --------------------------
# # Step 5: Output the Best Path and Cost
# # --------------------------
# print("Best path found:", best_path)
# print("Best cost:", best_cost)












# import networkx as nx
# import csv
# import random
# import math
# import matplotlib.pyplot as plt

# def load_graph_from_csv(filename):
#     """
#     Load graph from a CSV file.
#     CSV Format: source, target, weight
#     """
#     G = nx.Graph()
#     with open(filename, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             src = int(row['source'])
#             dst = int(row['target'])
#             weight = float(row['weight'])
#             G.add_edge(src, dst, weight=weight)
#     return G

# def initialize_pheromones(G, initial_pheromone=1.0):
#     """
#     Initialize pheromone on every edge in both directions.
#     """
#     pheromones = {}
#     for edge in G.edges():
#         pheromones[edge] = initial_pheromone
#         pheromones[(edge[1], edge[0])] = initial_pheromone  # undirected graph: mirror edge
#     return pheromones

# def select_next_node(G, pheromones, current_node, visited, alpha, beta):
#     """
#     Select next node based on a probability distribution influenced by pheromone level and heuristic (1/weight).
#     """
#     # List neighbors not visited yet
#     neighbors = [n for n in G.neighbors(current_node) if n not in visited]
#     if not neighbors:
#         return None

#     prob_list = []
#     total_prob = 0.0

#     # Calculate probabilities for each neighbor
#     for neighbor in neighbors:
#         edge = (current_node, neighbor)
#         tau = pheromones.get(edge, 1.0)
#         weight = G[current_node][neighbor]['weight']
#         heuristic = 1.0 / weight  # Higher if weight is lower
#         prob = (tau ** alpha) * (heuristic ** beta)
#         prob_list.append((neighbor, prob))
#         total_prob += prob

#     # Randomly select next node using roulette wheel selection
#     r = random.random() * total_prob
#     cumulative = 0.0
#     for neighbor, prob in prob_list:
#         cumulative += prob
#         if cumulative >= r:
#             return neighbor
#     return neighbors[-1]  # fallback

# def aco_network_routing(G, source, destination, num_ants=20, num_iterations=100, alpha=1.0, beta=2.0,
#                         evaporation_rate=0.5, Q=100):
#     """
#     Run the Ant Colony Optimization algorithm for network routing.
    
#     Returns the best path and its cost.
#     """
#     pheromones = initialize_pheromones(G, initial_pheromone=1.0)
#     best_path = None
#     best_cost = float('inf')
    
#     # Main ACO loop
#     for iteration in range(num_iterations):
#         all_paths = []
#         all_costs = []
        
#         for ant in range(num_ants):
#             path = [source]
#             current_node = source
            
#             # Build path until destination reached or ant is stuck
#             while current_node != destination:
#                 next_node = select_next_node(G, pheromones, current_node, path, alpha, beta)
#                 if next_node is None:
#                     path = None  # Dead end encountered
#                     break
#                 path.append(next_node)
#                 current_node = next_node
#                 # Prevent infinite loops
#                 if len(path) > len(G.nodes()):
#                     path = None
#                     break
            
#             # Evaluate path cost if a valid path was found
#             if path is not None:
#                 cost = sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
#                 all_paths.append(path)
#                 all_costs.append(cost)
#                 # Update best path if this one is better
#                 if cost < best_cost:
#                     best_cost = cost
#                     best_path = path

#         # Log iteration best cost
#         print(f"Iteration {iteration + 1}/{num_iterations} - Best Cost: {best_cost}")

#         # Pheromone evaporation (simulate the scent fading)
#         for edge in pheromones:
#             pheromones[edge] *= (1 - evaporation_rate)
        
#         # Pheromone update: reinforce good paths
#         for path, cost in zip(all_paths, all_costs):
#             pheromone_deposit = Q / cost  # Better paths deposit more pheromone
#             for i in range(len(path) - 1):
#                 edge = (path[i], path[i+1])
#                 # Update pheromones in both directions
#                 pheromones[edge] += pheromone_deposit
#                 pheromones[(edge[1], edge[0])] += pheromone_deposit

#     return best_path, best_cost

# def visualize_best_path(G, best_path):
#     """
#     Visualize the network and highlight the best path found.
#     """
#     pos = nx.spring_layout(G, seed=42)  # Compute layout
#     plt.figure(figsize=(10, 7))
    
#     # Draw the full graph in light gray
#     nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray', node_size=500)
    
#     # If a best path is found, highlight it
#     if best_path:
#         # Create a list of edge tuples from the best path
#         best_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path) - 1)]
#         nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2)
    
#     plt.title("Network Graph with Best Path Highlighted")
#     plt.show()

# # -------------- Main Execution --------------

# # Load the graph from CSV
# filename = 'network_edges.csv'
# G = load_graph_from_csv(filename)

# # Define source and destination nodes
# source = 0
# destination = max(G.nodes())  # Example: choose the highest numbered node

# # Run the ACO algorithm for network routing
# best_path, best_cost = aco_network_routing(G, source, destination, num_ants=20, num_iterations=100,
#                                            alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100)

# print("Final Best Path:", best_path)
# print("Final Best Cost:", best_cost)

# # Visualize the best path on the network graph
# visualize_best_path(G, best_path)







import networkx as nx
import csv
import random
import math
import matplotlib.pyplot as plt

def load_graph_from_csv(filename):
    """
    Load graph from a CSV file.
    CSV Format: source, target, weight
    """
    G = nx.Graph()
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['source'])
            dst = int(row['target'])
            weight = float(row['weight'])
            G.add_edge(src, dst, weight=weight)
    return G

def initialize_pheromones(G, initial_pheromone=1.0):
    """
    Initialize pheromone on every edge in both directions.
    """
    pheromones = {}
    for edge in G.edges():
        pheromones[edge] = initial_pheromone
        pheromones[(edge[1], edge[0])] = initial_pheromone  # undirected graph: mirror edge
    return pheromones


def cost_edge(G, i, j, traffic):
    """
    Calculate the cost of traversing edge (i,j) using a complex cost function.
    Here, the cost is defined as:
    
        cost(i,j) = base_weight * ((traffic[i] + traffic[j]) / 2)
    
    - base_weight: G[i][j]['weight']
    - traffic[node]: congestion factor (1.0 for normal, >1.0 if congested)
    """
    base_weight = G[i][j]['weight']
   
    traffic_i = traffic.get(i, 1.0)
    traffic_j = traffic.get(j, 1.0)
    avg_traffic = (traffic_i + traffic_j) / 2.0
    return base_weight * avg_traffic

def select_next_node(G, pheromones, current_node, visited, alpha, beta, traffic):
    """
    Select next node based on a probability distribution influenced by pheromone level and 
    a complex heuristic (the inverse of our cost function).
    """
    
    neighbors = [n for n in G.neighbors(current_node) if n not in visited]
    if not neighbors:
        return None

    prob_list = []
    total_prob = 0.0

   
    for neighbor in neighbors:
        edge = (current_node, neighbor)
        tau = pheromones.get(edge, 1.0)
       
        edge_cost = cost_edge(G, current_node, neighbor, traffic)
        heuristic = 1.0 / edge_cost  # Lower cost edges have higher heuristic value
        prob = (tau ** alpha) * (heuristic ** beta)
        prob_list.append((neighbor, prob))
        total_prob += prob

    
    r = random.random() * total_prob
    cumulative = 0.0
    for neighbor, prob in prob_list:
        cumulative += prob
        if cumulative >= r:
            return neighbor
    return neighbors[-1]  # fallback if roulette fails

def aco_network_routing(G, source, destination, traffic, num_ants=20, num_iterations=100, 
                        alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
    """
    Run the Ant Colony Optimization algorithm for network routing with a complex cost function.
    
    Returns the best path and its cost.
    """
    pheromones = initialize_pheromones(G, initial_pheromone=1.0)
    best_path = None
    best_cost = float('inf')
    
    # Main ACO loop
    for iteration in range(num_iterations):
        all_paths = []
        all_costs = []
        
        for ant in range(num_ants):
            path = [source]
            current_node = source
            
            
            while current_node != destination:
                next_node = select_next_node(G, pheromones, current_node, path, alpha, beta, traffic)
                if next_node is None:
                    path = None  # Dead end encountered
                    break
                path.append(next_node)
                current_node = next_node
                
                if len(path) > len(G.nodes()):
                    path = None
                    break
            
           
            if path is not None:
                
                total_path_cost = 0.0
                for i in range(len(path) - 1):
                    total_path_cost += cost_edge(G, path[i], path[i+1], traffic)
                all_paths.append(path)
                all_costs.append(total_path_cost)
               
                if total_path_cost < best_cost:
                    best_cost = total_path_cost
                    best_path = path

        
        print(f"Iteration {iteration + 1}/{num_iterations} - Best Cost: {best_cost}")

       
        for edge in pheromones:
            pheromones[edge] *= (1 - evaporation_rate)
        
        
        for path, p_cost in zip(all_paths, all_costs):
            pheromone_deposit = Q / p_cost  # Better (lower cost) paths deposit more pheromone
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                pheromones[edge] += pheromone_deposit
                pheromones[(edge[1], edge[0])] += pheromone_deposit  # Update both directions

    return best_path, best_cost

def visualize_best_path(G, best_path):
    """
    Visualize the network and highlight the best path found.
    """
    pos = nx.spring_layout(G, seed=42)  
    plt.figure(figsize=(10, 7))
    
    
    nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray', node_size=500)
    
   
    if best_path:
        best_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2)
    
    plt.title("Network Graph with Best Path Highlighted")
    plt.show()



# Load the graph from CSV
filename = 'network_edges.csv'
G = load_graph_from_csv(filename)

source = 0
destination = max(G.nodes())  # For example, choose the highest numbered node


traffic = {node: random.uniform(1.0, 2.0) for node in G.nodes()}

# Run the ACO algorithm for network routing using the complex cost function
best_path, best_cost = aco_network_routing(G, source, destination, traffic,
                                           num_ants=20, num_iterations=100,
                                           alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100)

print("Final Best Path:", best_path)
print("Final Best Cost:", best_cost)

# Visualize the best path on the network graph
visualize_best_path(G, best_path)











# import networkx as nx
# import csv
# import random
# import math
# import matplotlib.pyplot as plt
# import numpy as np

# def load_graph_from_csv(filename):
#     G = nx.Graph()
#     with open(filename, 'r') as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             src = int(row['source'])
#             dst = int(row['target'])
#             weight = float(row['weight'])
#             G.add_edge(src, dst, weight=weight)
#     return G

# def initialize_pheromones(G, initial_pheromone=1.0):
#     pheromones = {}
#     for edge in G.edges():
#         pheromones[edge] = initial_pheromone
#         pheromones[(edge[1], edge[0])] = initial_pheromone
#     return pheromones

# def cost_edge(G, i, j, traffic):
#     base_weight = G[i][j]['weight']
#     traffic_i = traffic.get(i, 1.0)
#     traffic_j = traffic.get(j, 1.0)
#     avg_traffic = (traffic_i + traffic_j) / 2.0
#     return base_weight * avg_traffic

# def select_next_node(G, pheromones, current_node, visited, alpha, beta, traffic):
#     neighbors = [n for n in G.neighbors(current_node) if n not in visited]
#     if not neighbors:
#         return None

#     prob_list = []
#     total_prob = 0.0
#     for neighbor in neighbors:
#         edge = (current_node, neighbor)
#         tau = pheromones.get(edge, 1.0)
#         heuristic = 1.0 / cost_edge(G, current_node, neighbor, traffic)
#         prob = (tau ** alpha) * (heuristic ** beta)
#         prob_list.append((neighbor, prob))
#         total_prob += prob

#     r = random.random() * total_prob
#     cumulative = 0.0
#     for neighbor, prob in prob_list:
#         cumulative += prob
#         if cumulative >= r:
#             return neighbor
#     return neighbors[-1]

# def aco_network_routing(G, source, destination, traffic, num_ants=20, num_iterations=50,
#                         alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
#     pheromones = initialize_pheromones(G, initial_pheromone=1.0)
#     best_path = None
#     best_cost = float('inf')

#     for _ in range(num_iterations):
#         all_paths = []
#         all_costs = []

#         for _ in range(num_ants):
#             path = [source]
#             current_node = source
#             while current_node != destination:
#                 next_node = select_next_node(G, pheromones, current_node, path, alpha, beta, traffic)
#                 if next_node is None:
#                     path = None
#                     break
#                 path.append(next_node)
#                 current_node = next_node
#                 if len(path) > len(G.nodes()):
#                     path = None
#                     break

#             if path is not None:
#                 total_path_cost = sum(cost_edge(G, path[i], path[i+1], traffic) for i in range(len(path) - 1))
#                 all_paths.append(path)
#                 all_costs.append(total_path_cost)
#                 if total_path_cost < best_cost:
#                     best_cost = total_path_cost
#                     best_path = path

#         for edge in pheromones:
#             pheromones[edge] *= (1 - evaporation_rate)

#         for path, p_cost in zip(all_paths, all_costs):
#             pheromone_deposit = Q / p_cost
#             for i in range(len(path) - 1):
#                 edge = (path[i], path[i+1])
#                 pheromones[edge] += pheromone_deposit
#                 pheromones[(edge[1], edge[0])] += pheromone_deposit

#     return best_path, best_cost

# def visualize_best_path(G, best_path):
#     pos = nx.spring_layout(G, seed=42)
#     plt.figure(figsize=(10, 7))
#     nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray', node_size=500)
#     if best_path:
#         best_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path) - 1)]
#         nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2)
#     plt.title("Network Graph with Best Path Highlighted")
#     plt.show()

# # --- Main Execution ---

# filename = 'network_edges.csv'
# G = load_graph_from_csv(filename)
# source = 0
# destination = max(G.nodes())
# traffic = {node: random.uniform(1.0, 2.0) for node in G.nodes()}

# # Generate 500 (alpha, beta) pairs
# alpha_beta_list = list({(round(random.uniform(0.1, 5.0), 2), round(random.uniform(0.1, 5.0), 2)) for _ in range(1000)})
# results = []
# global_best_cost = float('inf')
# global_best = None

# for i, (alpha, beta) in enumerate(alpha_beta_list[:500]):
#     print(f"[{i+1}/500] Testing alpha={alpha}, beta={beta}")
#     path, cost = aco_network_routing(
#         G, source, destination, traffic,
#         num_ants=20, num_iterations=50,
#         alpha=alpha, beta=beta, evaporation_rate=0.5, Q=100
#     )
#     results.append({"alpha": alpha, "beta": beta, "cost": cost, "path": path})

#     if cost < global_best_cost:
#         global_best_cost = cost
#         global_best = {"alpha": alpha, "beta": beta, "cost": cost, "path": path}

# # Save results to CSV
# with open('aco_500_alpha_beta_results.csv', 'w', newline='') as csvfile:
#     fieldnames = ['alpha', 'beta', 'cost', 'path']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#     for res in results:
#         writer.writerow({
#             'alpha': res['alpha'],
#             'beta': res['beta'],
#             'cost': res['cost'],
#             'path': ' -> '.join(map(str, res['path'])) if res['path'] else "None"
#         })

# print("\n=== Global Best Result ===")
# print(f"Alpha: {global_best['alpha']}, Beta: {global_best['beta']}, Cost: {global_best['cost']}")
# print(f"Path: {' -> '.join(map(str, global_best['path']))}")

# # Optional: Visualize best path
# visualize_best_path(G, global_best['path'])