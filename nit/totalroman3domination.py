import time

def total_roman_3_domination(adj_list):
    def is_valid(labeling):
        for v in adj_list:
            if labeling[v] == 0 and not any(labeling[neighbor] == 3 for neighbor in adj_list[v]):
                return False
            if labeling[v] == 1:
                neighbor_sum = sum(labeling[neighbor] for neighbor in adj_list[v])
                if neighbor_sum < 2:
                    return False
            if labeling[v] > 0 and all(labeling[neighbor] == 0 for neighbor in adj_list[v]):
                return False
        return True

    def dfs(v, labeling, best_labeling, partial_cost):
        if v == len(vertices):
            if is_valid(labeling):
                total_cost = sum(labeling[v] for v in vertices)
                best_cost = sum(best_labeling[v] for v in vertices) if best_labeling else float('inf')
                if total_cost < best_cost:
                    for i in range(len(vertices)):
                        best_labeling[vertices[i]] = labeling[vertices[i]]
            return

        current_vertex = vertices[v]

        # Try label 3 first
        labeling[current_vertex] = 3
        if is_valid_partial(labeling, current_vertex):
            dfs(v + 1, labeling, best_labeling, partial_cost + 3)

        # Try label 1 next
        labeling[current_vertex] = 1
        if is_valid_partial(labeling, current_vertex):
            dfs(v + 1, labeling, best_labeling, partial_cost + 1)

        # Try label 0 last
        labeling[current_vertex] = 0
        if is_valid_partial(labeling, current_vertex):
            dfs(v + 1, labeling, best_labeling, partial_cost)

        # Reset the label for backtracking
        del labeling[current_vertex]

    def is_valid_partial(labeling, current_vertex):
        if labeling[current_vertex] == 0:
            for neighbor in adj_list[current_vertex]:
                if labeling.get(neighbor) == 3:
                    return True
            return False
        if labeling[current_vertex] == 1:
            neighbor_sum = sum(labeling.get(neighbor, 0) for neighbor in adj_list[current_vertex])
            unknown_neighbors = [n for n in adj_list[current_vertex] if n not in labeling]
            if neighbor_sum >= 2:
                return True
            elif neighbor_sum == 1 and unknown_neighbors:
                return True
            elif not unknown_neighbors:
                return False
        if labeling[current_vertex] == 3:
            for neighbor in adj_list[current_vertex]:
                if labeling.get(neighbor, 0) > 0:
                    return True
            if any(neighbor not in labeling for neighbor in adj_list[current_vertex]):
                return True
            return False
        return True

    vertices = list(adj_list.keys())
    best_labeling = {}
    labeling = {}  # Start with unassigned vertices

    dfs(0, labeling, best_labeling, 0)

    if not best_labeling:
        return None
    return best_labeling

# Read number of vertices and edges
num_vertices, num_edges = map(int, input("Enter the number of vertices and edges: ").split())

# Initialize an empty adjacency list
adj_list = {i: [] for i in range(1, num_vertices + 1)}  # Create vertices 1, 2, 3, ...

# Read the edges
print("enter edges line by line")
for _ in range(num_edges):
    u, v = map(int, input().split())
    if u != v:  # Ignore self-loops
        adj_list[u].append(v)
        adj_list[v].append(u)

# Measure the execution time
start_time = time.time()

# Call the function to find the Total Roman {3}-Domination labeling
rdf_labels = total_roman_3_domination(adj_list)

end_time = time.time()

# Calculate the total execution time
execution_time = end_time - start_time

# Print the adjacency list (for debugging purposes)
print("The adjacency list is", adj_list)

# Print the result
if rdf_labels is None:
    print("No valid Total Roman {3}-Domination set exists for this graph.")
else:
    print("Total Roman {3}-Domination labeling:")
    for vertex, label in sorted(rdf_labels.items()):
        print(f"{vertex}: {label}")

    # Calculate and print the total cost
    total_cost = sum(rdf_labels.values())
    print(f"\nTotal cost: {total_cost}")

# Print the execution time
print(f"\nExecution time: {execution_time} seconds")
