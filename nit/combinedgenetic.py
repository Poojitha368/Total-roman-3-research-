import random

def heuristic1(graph):
    labeling = {}
    vertices = list(graph.keys())
    unprocessed_vertices = set(vertices)  # Track unprocessed vertices

    while unprocessed_vertices:
        vertex = random.choice(list(unprocessed_vertices))  # Choose from unprocessed vertices
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

        unprocessed_vertices.remove(vertex)  # Remove the processed vertex

    # Ensure all vertices are labeled
    for vertex in vertices:
        if vertex not in labeling:
            labeling[vertex] = 0  # Or any default label you prefer

    # Calculate the sum of the labels
    sum_labels = sum(labeling.values())
    return labeling, sum_labels

def heuristic2(graph):
    num_vertices = len(graph)
    # Initialize all node labels to -1 (unlabeled)
    labels = {node: -1 for node in graph.keys()}
    
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
    for vertex in graph.keys():
        if labels[vertex] == 2:
            neighbors = graph.get(vertex, [])
            if not any(labels.get(neighbor, -1) == 1 for neighbor in neighbors):
                if neighbors:
                    random_neighbor = random.choice(neighbors)
                    labels[random_neighbor] = 1

    # Label isolated vertices as 1
    for vertex in graph.keys():
        if labels[vertex] == -1:
            labels[vertex] = 1
   
    # Calculate the sum of the labels
    sum_labels = sum(labels.values())
    return labels, sum_labels

def is_feasible(chromosome, graph):
    labeling = chromosome.copy()

    for vertex in graph:
        if vertex not in labeling:
            labeling[vertex] = 0  # Assign default label if not present

    # First loop: handle the label assignment based on the current labels
    for vertex in labeling:
        label = labeling[vertex]
        neighbors = graph[vertex]
        neighbor_labels = [labeling[neighbor] for neighbor in neighbors]

        # If any vertex with label 1, 2, or 3 is surrounded by all label 0 neighbors, mark neighbors for update
        if label in [1, 2, 3] and all(neighbor_label == 0 for neighbor_label in neighbor_labels):
            for neighbor in neighbors:
                labeling[neighbor] = 1  # Temporary assignment to avoid conflict

        # If any vertex has exactly two neighbors, ensure both neighbors' labels are 1
        if len(neighbors) == 2:
            for neighbor in neighbors:
                if len(graph[neighbor]) == 1:  # Check if neighbor has exactly one neighbor
                    labeling[neighbor] = 2  # Temporary assignment to avoid conflict
                else:
                    labeling[neighbor] = 1  # Temporary assignment to avoid conflict

    # Second loop: make additional adjustments based on updated labels
    for vertex in labeling:
        label = labeling[vertex]
        neighbors = graph[vertex]
        neighbor_labels = [labeling[neighbor] for neighbor in neighbors]
        sum_neighbor_labels = sum(neighbor_labels)

        if label == 0 and sum_neighbor_labels < 3:
            random_neighbor = random.choice(neighbors)
            labeling[random_neighbor] = 3

        elif label == 1 and sum_neighbor_labels < 2:
            random_neighbor = random.choice(neighbors)
            labeling[random_neighbor] = 2

    return labeling

def initialize_population_with_heuristic(population_size, graph):
    population = []
    for _ in range(population_size // 2):
        heuristic_labeling1, _ = heuristic1(graph)
        population.append(heuristic_labeling1)
    
    for _ in range(population_size // 2):
        heuristic_labeling2, _ = heuristic2(graph)
        population.append(heuristic_labeling2)

    return population

def fitness(chromosome):
    return sum(chromosome.values())

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = {vertex: (parent1[vertex] if i < crossover_point else parent2[vertex]) 
             for i, vertex in enumerate(parent1)}
    return child

def mutate(chromosome, mutation_rate):
    mutated_chromosome = chromosome.copy()
    for vertex in mutated_chromosome:
        if random.random() < mutation_rate:
            mutated_chromosome[vertex] = random.randint(0, 3)
    return mutated_chromosome

def select_parents(population):
    fitness_values = [fitness(chromosome) for chromosome in population]
    total_fitness = sum(fitness_values)
    
    if total_fitness == 0:
        probabilities = [1 / len(population)] * len(population)
    else:
        probabilities = [fit / total_fitness for fit in fitness_values]
    
    parent1 = random.choices(population, weights=probabilities)[0]
    parent2 = random.choices(population, weights=probabilities)[0]
    
    return parent1, parent2

def genetic_algorithm(num_vertices, graph, population_size=1000, mutation_rate=0.1, generations=1000):
    population = initialize_population_with_heuristic(population_size, graph)
    
    best_chromosome = None
    best_fitness_value = float('inf')

    for generation in range(generations):
        new_population = []

        # Add elitism: retain the top 10% of the population
        sorted_population = sorted(population, key=fitness)
        elite_count = max(1, population_size // 10)
        new_population.extend(sorted_population[:elite_count])

        attempts = 0
        max_attempts = population_size * 10  # limit attempts to prevent infinite loops

        while len(new_population) < population_size and attempts < max_attempts:
            parent1, parent2 = select_parents(population)
            offspring1 = crossover(parent1, parent2)
            offspring2 = crossover(parent2, parent1)
            mutated_offspring1 = mutate(offspring1, mutation_rate)
            mutated_offspring2 = mutate(offspring2, mutation_rate)
            
            feasible1 = is_feasible(mutated_offspring1, graph)
            new_population.append(feasible1)
            feasible2 = is_feasible(mutated_offspring2, graph)
            new_population.append(feasible2)

            attempts += 1

        population = new_population
        
        # Find the best solution in the current population
        for chromosome in population:
            current_fitness = fitness(chromosome)
            if current_fitness < best_fitness_value:
                best_fitness_value = current_fitness
                best_chromosome = chromosome
        
        print(f"Generation {generation + 1}: Best fitness = {best_fitness_value}")

        if best_fitness_value == 0:
            break  # Solution found

    return best_chromosome, best_fitness_value

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
    # print("Enter the number of vertices and edges:")
    num_vertices, num_edges = map(int, input().split())

    # print("Enter the edges (two numbers separated by space) line by line:")
    edges = [tuple(map(int, input().split())) for _ in range(num_edges)]

    graph = parse_graph(num_vertices, num_edges, edges)

    best_chromosome, best_fitness_value = genetic_algorithm(num_vertices, graph, population_size=100, generations=1000)

    print("Best Labeling:")
    print(best_chromosome)
    print(f"Best Sum of labels: {best_fitness_value}")

if __name__ == "__main__":
    main()