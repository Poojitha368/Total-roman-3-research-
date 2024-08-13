import random

def heuristic1(graph):
    labeling = {}
    vertices = list(graph.keys())
    unprocessed_vertices = set(vertices)  # Track unprocessed vertices

    while unprocessed_vertices:
        # Pick a vertex randomly from unprocessed vertices
        vertex = random.choice(list(unprocessed_vertices))

        if vertex not in labeling:
            neighbors = graph[vertex]
            labeled_neighbors = [neighbor for neighbor in neighbors if neighbor in labeling]

            if len(neighbors) == 2:  # If vertex has exactly two neighbors
                labeling[vertex] = 1
                for neighbor in neighbors:
                    if neighbor not in labeling:
                        labeling[neighbor] = 1

                        # Check if neighbor has more than one neighbor
                        if len(graph[neighbor]) <= 1:
                            labeling[neighbor] = 2
            else:
                if labeled_neighbors and all(labeling[neighbor] == 1 for neighbor in labeled_neighbors):
                    labeling[vertex] = 2

                    # Ensure only one neighbor has label '1'
                    if sum(1 for neighbor in labeled_neighbors if labeling[neighbor] == 1) != 1:
                        for neighbor in labeled_neighbors:
                            if labeling[neighbor] == 0:
                                labeling[neighbor] = 1
                                break
                else:
                    labeling[vertex] = 3

                    # Assign label '0' to its neighbors
                    for neighbor in neighbors:
                        if neighbor not in labeling:
                            labeling[neighbor] = 0

                    # Ensure at least one neighbor is labeled '1'
                    if sum(1 for neighbor in neighbors if labeling.get(neighbor) == 1) == 0:
                        for neighbor in neighbors:
                            if labeling[neighbor] == 0:
                                labeling[neighbor] = 1
                                break

        unprocessed_vertices.remove(vertex)  # Remove the processed vertex

    # Ensure all vertices are labeled
    for vertex in vertices:
        if vertex not in labeling:
            labeling[vertex] = 0  # Or any default label you prefer

    # Calculate the sum of the labels
    sum_labels = sum(labeling.values())
    return labeling, sum_labels

def parse_graph(num_vertices, num_edges, edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        graph[v].append(u)
    return graph

def main():
    print("Enter the number of vertices and edges:")
    num_vertices, num_edges = map(int, input().split())

    print("Enter the edges (two numbers separated by space) line by line:")
    edges = [tuple(map(int, input().split())) for _ in range(num_edges)]

    graph = parse_graph(num_vertices, num_edges, edges)

    # Initialize variables to track the best result
    best_sum = None
    best_labeling = None

    # Run the heuristic function 1000 times
    for i in range(1000):
        labeling, sum_labels = heuristic1(graph)

        # Update best result if this run is better
        if best_sum is None or sum_labels < best_sum:
            best_sum = sum_labels
            best_labeling = labeling

    # Print the best labeling and sum of labels found
    print("Best Labeling:")
    print(best_labeling)
    print(f"Best Sum of labels: {best_sum}")

if __name__ == "__main__":
    main()