def calculate_upper_bound(vertices, edges, edge_list):
    # Calculate minimum degree
    degree_count = [0] * (vertices + 1)  # +1 because vertices are 1-indexed
    for edge in edge_list:
        degree_count[edge[0]] += 1
        degree_count[edge[1]] += 1
    
    min_degree = min(degree_count[1:])  # Ignore degree_count[0], as vertices start from 1
    
    # Calculate upper bound
    upper_bound = 2 * vertices - min_degree
    
    return upper_bound

def main():
    # Input graph data
    vertices, edges = map(int, input().split())
    
    edge_list = []
    # print("enter the edge line by line: ")
    for _ in range(edges):
        edge = tuple(map(int, input().split()))
        edge_list.append(edge)
    
    # Calculate upper bound
    upper_bound = calculate_upper_bound(vertices, edges, edge_list)
    
    # Output upper bound
    print(f"Upper bound for the graph is: {upper_bound}")

if __name__ == "__main__":
    main()
