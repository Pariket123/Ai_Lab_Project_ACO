We have implemeted ACO on Network Routing
Network Routing Optimization is about finding the best path for data (or information) to travel from a source node to a destination node in a network. In real-world networks—such as the internet, 
transportation grids, or even social networks—each connection (or edge) has a "cost" (e.g., delay, congestion, distance, or bandwidth usage).
The goal is to minimize the overall cost by selecting an optimal route.

Key Concepts in Routing:
Nodes: Represent routers, switches, or intersections in a network.


Edges: Represent connections between nodes, each with a weight (cost).


Cost Metrics: Could be time delay, distance, congestion level, etc.


Optimal Path: The route that minimizes (or sometimes maximizes, depending on the criteria) the total cost from start to finish.

- weight: Base cost (distance, for example) 
- latency: Communication or travel delay 
- reliability: Quality factor (0 to 1; lower values produce higher penalty)
 - energy: Energy consumption for the edge
 - traffic: Congestion factor on nodes (average of node i and node j)
The cost is computed as a weighted sum: cost(i, j) = w1 * weight + w2 * latency + w3 * reliability_penalty + w4 * energy + w5 * avg_traffic where reliability_penalty = 1 / reliability (if reliability > 0) and avg_traffic = (traffic[i] + traffic[j]) / 2. 
Adjust w1...w5 to change the importance of each factor.
