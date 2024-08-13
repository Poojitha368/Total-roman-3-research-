import random

def heuristic2(graph, num_vertices):
    # Initialize all node labels to -1 (unlabeled)
    labels = {node: -1 for node in range(1, num_vertices + 1)}
    
    # Set to keep track of visited vertices
    visited = set()

    while True:
        # Find the highest degree vertex that is still unlabeled
        high_degree_vertex = None
        max_degree = -1
        highest_degree_vertices = []

        for node in graph.keys():
            if labels[node] == -1 and node not in visited:
                degree = len(graph[node])
                if degree > max_degree:
                    max_degree = degree
                    highest_degree_vertices = [node]
                elif degree == max_degree:
                    highest_degree_vertices.append(node)

        if not highest_degree_vertices:
            break  # No more vertices to label

        # If there are multiple vertices with the same highest degree, choose one randomly
        if len(highest_degree_vertices) > 1:
            high_degree_vertex = random.choice(highest_degree_vertices)
        else:
            high_degree_vertex = highest_degree_vertices[0]

        # Collect its neighbors
        neighbors = graph.get(high_degree_vertex, [])  # Use .get() to handle potential KeyError

        # Check if all neighbors are visited or labeled
        all_neighbors_visited = all(neighbor in visited or labels.get(neighbor, -1) != -1 for neighbor in neighbors)

        if all_neighbors_visited:
            # Label the vertex as 2 if all neighbors are labeled
            labels[high_degree_vertex] = 2
        else:
            # Label the vertex as 3
            labels[high_degree_vertex] = 3

            # Collect unlabeled neighbors
            neighbors_labeled = [neighbor for neighbor in neighbors if labels[neighbor] == -1]

            # If there are neighbors, label one randomly as 1 and the rest as 0
            if neighbors_labeled:
                # Choose one random neighbor to label as 1
                random.shuffle(neighbors_labeled)  # Shuffle to randomize the choice
                labels[neighbors_labeled[0]] = 1  # Label the first as 1

                # Label the remaining neighbors as 0
                for neighbor in neighbors_labeled[1:]:
                    labels[neighbor] = 0

        # Mark this vertex as visited
        visited.add(high_degree_vertex)

    # Ensure vertices labeled as 2 have exactly one neighbor labeled as 1
    for vertex in range(1, num_vertices + 1):
        if labels[vertex] == 2:
            neighbors = graph.get(vertex, [])
            if not any(labels.get(neighbor, -1) == 1 for neighbor in neighbors):
                if neighbors:
                    random_neighbor = random.choice(neighbors)
                    labels[random_neighbor] = 1

    # Label isolated vertices as 1
    for vertex in range(1, num_vertices + 1):
        if labels[vertex] == -1:
            labels[vertex] = 1

    return labels

def parse_edges(num_edges):
    edges = []
    print(f"Enter {num_edges} edges (two numbers separated by space) line by line:")
    for i in range(1, num_edges + 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    return edges

def create_graph(edges):
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
    edges = parse_edges(num_edges)
    
    # Create the adjacency list from the edges
    graph = create_graph(edges)

    # Get the labeling of vertices using heuristic2 function
    labels = heuristic2(graph, num_vertices)
    print("Final labeling dictionary:", labels)
    total_cost = sum(labels.values())
    print("Total cost of labeling:", total_cost)

if __name__ == "__main__":
    main()
