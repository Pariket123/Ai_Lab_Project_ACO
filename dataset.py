import networkx as nx
import random
import csv

num_nodes = 50       # Total number of nodes (routers)
edge_prob = 0.1      # Probability that an edge exists between any two nodes
min_weight = 1       # Minimum possible weight (e.g., delay or cost)
max_weight = 10      # Maximum possible weight

G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=42, directed=False)

for u, v in G.edges():
    G[u][v]['weight'] = random.randint(min_weight, max_weight)

csv_filename = "network_edges.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["source", "target", "weight"]) 
    for u, v, data in G.edges(data=True):
        writer.writerow([u, v, data['weight']])

print(f"Synthetic network dataset saved as '{csv_filename}'")
