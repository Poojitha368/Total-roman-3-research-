''' Heuristic 1 -> We will take the random vertex and label it 3 and its neighbours as 
0 and check if no two 0's are adjacent and 3 have atleast one label 1 neighbour  '''

import random

def assign_labels(graph):
    labeling = {}
    vertices = list(graph.keys())

    while vertices:
        # Choose a random vertex and assign label '3'
        vertex = random.choice(vertices)
        print(vertex)
        labeling[vertex] = 3
        neighbors = graph[vertex]

        # Assign label '0' to its neighbors
        for neighbor in neighbors:
            if neighbor not in labeling:
                labeling[neighbor] = 0

        # Count the number of neighbors labeled '1'
        count_label_1 = sum(1 for neighbor in neighbors if labeling.get(neighbor) == 1)

        # Ensure only one neighbor is labeled as '1'
        labeled_as_one = False
        for neighbor in neighbors:
            if labeling[neighbor] == 0:
                if count_label_1 == 0:
                    labeling[neighbor] = 1
                    labeled_as_one = True
                break

        # If no neighbor was labeled as '1', choose one randomly and label it as '1'
        if not labeled_as_one and neighbors:
            neighbor = random.choice(neighbors)
            labeling[neighbor] = 1

        # Remove labeled vertices from the list of vertices to process
        vertices = [v for v in vertices if v not in labeling]

    return labeling

def parse_graph(num_vertices, num_edges, edges):
    graph = {}
    for edge in edges:
        u, v = edge
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    return graph

# Get input graph from the user
print("Enter the number of vertices and edges:")
num_vertices, num_edges = map(int, input().split())

print("Enter the edges (two numbers separated by space) line by line:")
edges = [tuple(map(int, input().split())) for _ in range(num_edges)]

graph = parse_graph(num_vertices, num_edges, edges)

# Get the labeling of vertices
labeling = assign_labels(graph)
print(labeling)
