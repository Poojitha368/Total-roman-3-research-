import random

def heuristic1(graph):
    labeling = {}
    vertices = list(graph.keys())
    unprocessed_vertices = set(vertices)  # Track unprocessed vertices

    while unprocessed_vertices:
        vertex = random.choice(list(unprocessed_vertices))  # Choose from unprocessed vertices
        print(vertex)
        if vertex not in labeling:
            neighbors = graph[vertex]
            labeled_neighbors = [neighbor for neighbor in neighbors if neighbor in labeling]

            if labeled_neighbors and all(labeling[neighbor] for neighbor in labeled_neighbors):
                labeling[vertex] = 2

                # Ensure only one neighbor has label '1'
                count_label_1 = sum(1 for neighbor in labeled_neighbors if labeling[neighbor] == 1)
                if count_label_1 != 1:
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
                count_label_1 = sum(1 for neighbor in neighbors if labeling.get(neighbor) == 1)
                if count_label_1 == 0:
                    for neighbor in neighbors:
                        if labeling[neighbor] == 0:
                            labeling[neighbor] = 1
                            break

        unprocessed_vertices.remove(vertex)  # Remove the processed vertex

        # Ensure all vertices are labeled
    for vertex in vertices:
        if vertex not in labeling:
            labeling[vertex] = 0  # Or any default label you prefer

    return labeling

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

    # Get the labeling of vertices
    labeling = heuristic1(graph)
    print("Final labeling dictionary:")
    print(labeling)
    sum_labels = sum(labeling.values())
    print("Total cost of labeling",sum_labels)

if __name__ == "__main__":
    main()
