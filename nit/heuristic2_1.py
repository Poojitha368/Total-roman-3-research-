''' No 2 in this only 3,0,1 and same high degree will choose the random vertex'''

import random

def label_graph(edges, num_vertices):
    # Create an adjacency list from the edges
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)

    # Initialize all node labels to -1 (unlabeled)
    labels = {node: -1 for node in range(1, num_vertices + 1)}

    while True:
        # Find the highest degree vertex that is still unlabeled
        high_degree_vertex = None
        highest_degree_vertices = []
        max_degree = -1
        for node in graph.keys():
            if labels[node] == -1:  # Check if the node is unlabeled
                degree = len(graph[node])
                if degree > max_degree:
                    max_degree = degree
                    highest_degree_vertices = [node]
                    high_degree_vertex = node
                elif degree == max_degree:
                    highest_degree_vertices.append(node)

        if high_degree_vertex is None:
            break  # No more vertices to label

        # If there are multiple vertices with the same highest degree, choose one randomly
        if len(highest_degree_vertices) > 1:
            high_degree_vertex = random.choice(highest_degree_vertices)

        # Label the highest degree vertex as 3
        labels[high_degree_vertex] = 3
        print(high_degree_vertex)

        # Collect its unlabeled neighbors
        neighbors_labeled = []
        for neighbor in graph[high_degree_vertex]:
            if labels[neighbor] == -1:  # Only label if it is still unlabeled
                neighbors_labeled.append(neighbor)

        num_neighbors = len(neighbors_labeled)

        # If there are neighbors, label them
        if num_neighbors > 0:
            # Sort neighbors by the number of their own connections (in descending order)
            neighbors_labeled.sort(key=lambda x: len(graph[x]), reverse=True)

            # Determine the number of neighbors to be labeled as 1
            num_ones = max(1, num_neighbors - 1)  # Minimum of 1, or n-1 if more neighbors

            # Label the required number of neighbors as 1
            for i in range(num_ones):
                labels[neighbors_labeled[i]] = 1

            # Label the remaining neighbors as 0
            for i in range(num_ones, num_neighbors):
                labels[neighbors_labeled[i]] = 0

            # Ensure no two 0s are adjacent
            for neighbor in neighbors_labeled:
                if labels[neighbor] == 0:
                    for adj in graph[neighbor]:
                        if labels[adj] == 0:
                            labels[adj] = 1

        # Remove the labeled vertex and its neighbors from future consideration
        del graph[high_degree_vertex]
        for neighbor in neighbors_labeled:
            if neighbor in graph:
                del graph[neighbor]
            for node, adj_list in graph.items():
                if neighbor in adj_list:
                    adj_list.remove(neighbor)

    # Label isolated vertices as 1
    for vertex in range(1, num_vertices + 1):
        if labels[vertex] == -1:
            labels[vertex] = 1

    return labels

def parse_edges(num_edges):
    edges = []
    print("Enter the edges (two numbers separated by space) line by line:")
    for _ in range(num_edges):
        edge = tuple(map(int, input().split()))
        edges.append(edge)
    return edges

# Get input from the user
print("Enter the number of vertices and edges:")
num_vertices, num_edges = map(int, input().split())

edges = parse_edges(num_edges)

# Get the labeling of vertices
labels = label_graph(edges, num_vertices)
print(labels)
