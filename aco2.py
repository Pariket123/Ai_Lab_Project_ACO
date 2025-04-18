import networkx as nx
import csv
import random
import math
import matplotlib.pyplot as plt

def load_graph_from_csv(filename):
    """
    Load graph from a CSV file with columns:
    source, target, weight, latency, reliability, energy
    """
    G = nx.Graph()
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row['source'])
            dst = int(row['target'])
            weight = float(row['weight'])
            latency = float(row['latency'])
            reliability = float(row['reliability'])
            energy = float(row['energy'])
            # Save all features as edge attributes
            G.add_edge(src, dst, weight=weight, latency=latency, reliability=reliability, energy=energy)
    return G


def initialize_pheromones(G, initial_pheromone=1.0):
    """
    Initialize pheromone on each edge (both directions, for an undirected graph).
    """
    pheromones = {}
    for edge in G.edges():
        pheromones[edge] = initial_pheromone
        pheromones[(edge[1], edge[0])] = initial_pheromone
    return pheromones



def realistic_cost_edge(G, i, j, traffic):
    """
    Compute a realistic cost for traversing the edge (i, j) using multiple factors:
    
    Features used:
      - weight: Base cost (distance, for example)
      - latency: Communication or travel delay
      - reliability: Quality factor (0 to 1; lower values produce higher penalty)
      - energy: Energy consumption for the edge
      - traffic: Congestion factor on nodes (average of node i and node j)
    
    The cost is computed as a weighted sum:
      cost(i, j) = w1 * weight + w2 * latency + w3 * reliability_penalty + w4 * energy + w5 * avg_traffic
    
    where reliability_penalty = 1 / reliability (if reliability > 0)
    and avg_traffic = (traffic[i] + traffic[j]) / 2.
    
    Adjust w1...w5 to change the importance of each factor.
    """
    # Retrieve edge feature values
    edge_attr = G[i][j]
    weight = edge_attr.get('weight', 1.0)
    latency = edge_attr.get('latency', 1.0)
    reliability = edge_attr.get('reliability', 1.0)
    energy = edge_attr.get('energy', 1.0)
    
    # Get traffic for node i and j (default to 1.0 if not provided)
    traffic_i = traffic.get(i, 1.0)
    traffic_j = traffic.get(j, 1.0)
    avg_traffic = (traffic_i + traffic_j) / 2.0
    
    
    reliability_penalty = 1.0 / reliability if reliability > 0 else float('inf')
    
    w1 = 0.5   # importance of weight
    w2 = 1.0   # importance of latency
    w3 = 2.0   # importance of reliability (penalty)
    w4 = 0.3   # importance of energy consumption
    w5 = 1.0   # importance of traffic congestion
    
    cost = (w1 * weight) + (w2 * latency) + (w3 * reliability_penalty) + (w4 * energy) + (w5 * avg_traffic)
    return cost



def select_next_node(G, pheromones, current_node, visited, alpha, beta, traffic):
    """
    Select the next node to visit from current_node using a probability distribution that
    depends on:
      - Pheromone level: tau^(alpha)
      - Heuristic value: (1 / realistic_cost_edge)^(beta)
    
    Only considers neighbors that have not been visited yet.
    """
   
    neighbors = [n for n in G.neighbors(current_node) if n not in visited]
    if not neighbors:
        return None

    prob_list = []
    total_prob = 0.0
    
    for neighbor in neighbors:
        edge = (current_node, neighbor)
        tau = pheromones.get(edge, 1.0)
        cost_val = realistic_cost_edge(G, current_node, neighbor, traffic)
        heuristic = 1.0 / cost_val if cost_val != 0 else float('inf')
        prob = (tau ** alpha) * (heuristic ** beta)
        prob_list.append((neighbor, prob))
        total_prob += prob
    
    # Roulette wheel selection
    r = random.random() * total_prob
    cumulative = 0.0
    for neighbor, prob in prob_list:
        cumulative += prob
        if cumulative >= r:
            return neighbor
    return neighbors[-1]  # Fallback



def aco_network_routing(G, source, destination, traffic, num_ants=20, num_iterations=50,
                        alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100):
    """
    Run the Ant Colony Optimization algorithm on the graph G from source to destination.
    The pheromone update is based on the cost of the path found.
    The cost is computed using the realistic_cost_edge function.
    
    The algorithm:
      - Initializes pheromones.
      - For a number of iterations, simulates a number of ants that build paths.
      - Updates pheromones based on the quality (cost) of paths.
      - Returns the best path found and its cost.
    """
    pheromones = initialize_pheromones(G, initial_pheromone=1.0)
    best_path = None
    best_cost = float('inf')
    
    for iteration in range(num_iterations):
        all_paths = []
        all_costs = []
        
        for ant in range(num_ants):
            path = [source]
            current_node = source
            
            # Build a path until we reach the destination
            while current_node != destination:
                next_node = select_next_node(G, pheromones, current_node, path, alpha, beta, traffic)
                if next_node is None:
                    path = None  # dead end encountered
                    break
                path.append(next_node)
                current_node = next_node
                
                if len(path) > len(G.nodes()):
                    path = None
                    break
            
            # If a valid path was found, compute its total cost.
            if path is not None:
                total_path_cost = sum(realistic_cost_edge(G, path[i], path[i+1], traffic)
                                       for i in range(len(path) - 1))
                all_paths.append(path)
                all_costs.append(total_path_cost)
                if total_path_cost < best_cost:
                    best_cost = total_path_cost
                    best_path = path

    
        for edge in pheromones:
            pheromones[edge] *= (1 - evaporation_rate)
        
      
        for path, p_cost in zip(all_paths, all_costs):
            pheromone_deposit = Q / p_cost
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                pheromones[edge] += pheromone_deposit
                pheromones[(path[i+1], path[i])] += pheromone_deposit
                
        print(f"Iteration {iteration+1}/{num_iterations} - Best Cost so far: {best_cost}")

    return best_path, best_cost



def visualize_best_path(G, best_path):
    """
    Plot the network graph and highlight the best path found.
    """
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color='gray', node_size=500)
    if best_path:
        best_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=best_edges, edge_color='red', width=2)
    plt.title("Network Graph with Best Path Highlighted")
    plt.show()


if __name__ == "__main__":
    # Load dataset
    filename = 'dataset.csv'
    G = load_graph_from_csv(filename)
    
    # Define source and destination nodes:
    # source = 0 and destination = highest numbered node in the graph
    source = 0
    destination = max(G.nodes())
    
    # Generate a traffic dictionary for nodes.
    traffic = {node: random.uniform(1.0, 2.0) for node in G.nodes()}
    
    best_path, best_cost = aco_network_routing(
        G, source, destination, traffic,
        num_ants=20, num_iterations=50,
        alpha=1.0, beta=2.0, evaporation_rate=0.5, Q=100
    )
    
    print("\n=== Final Best Path ===")
    print(f"Path: {' -> '.join(map(str, best_path)) if best_path else 'None'}")
    print(f"Cost: {best_cost}")
    
    # Visualize the best path on the graph.
    visualize_best_path(G, best_path)
