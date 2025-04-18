# data_set.py

import csv
import random

def generate_random_graph(num_nodes=20, num_edges=40, seed=42):
    random.seed(seed)
    
    # To avoid duplicate edges, we'll store pairs in a set.
    edges = set()
    
    while len(edges) < num_edges:
        # Randomly choose two distinct nodes
        src = random.randint(0, num_nodes - 1)
        dst = random.randint(0, num_nodes - 1)
        if src == dst:
            continue
        
        # Since this is an undirected graph, we can order the tuple (min, max)
        edge = (min(src, dst), max(src, dst))
        edges.add(edge)
    
    return list(edges)

def generate_edge_features():
    """
    Generate random features for an edge:
      - weight: base cost (e.g., distance) between 1 and 20
      - latency: between 10 and 100 milliseconds (ms)
      - reliability: between 0.70 and 1.00 (1 is best)
      - energy: energy consumption, between 5 and 50 units
    """
    weight = round(random.uniform(1, 20), 2)
    latency = round(random.uniform(10, 100), 2)
    reliability = round(random.uniform(0.70, 1.0), 2)
    energy = round(random.uniform(5, 50), 2)
    
    return weight, latency, reliability, energy

def save_graph_dataset_csv(filename="dataset.csv", num_nodes=20, num_edges=40):
    edges = generate_random_graph(num_nodes=num_nodes, num_edges=num_edges)
    
    with open(filename, mode="w", newline="") as csvfile:
        fieldnames = ['source', 'target', 'weight', 'latency', 'reliability', 'energy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for src, dst in edges:
            weight, latency, reliability, energy = generate_edge_features()
            writer.writerow({
                'source': src,
                'target': dst,
                'weight': weight,
                'latency': latency,
                'reliability': reliability,
                'energy': energy
            })
if __name__ == "__main__":
    # Increase to 100 nodes and 500 edges
    save_graph_dataset_csv(filename="large_dataset.csv", num_nodes=100, num_edges=500)
    print("Large dataset generated and saved as 'large_dataset.csv'.")

