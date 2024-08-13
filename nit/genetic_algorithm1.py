import random

def heuristic1(graph):
    labeling = {}
    vertices = list(graph.keys())
    unprocessed_vertices = set(vertices)  # Track unprocessed vertices

    while unprocessed_vertices:
        vertex = random.choice(list(unprocessed_vertices))  # Choose from unprocessed vertices
        # print(vertex)
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

    # Calculate the sum of the labels
    sum_labels = sum(labeling.values())
    return labeling, sum_labels


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
    for _ in range(population_size):
        heuristic_labeling, _ = heuristic1(graph)
        population.append(heuristic_labeling)
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

def genetic_algorithm(num_vertices, graph, population_size=100, mutation_rate=0.1, generations=1000):
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
            # print(mutated_offspring1,mutated_offspring2)
            
            feasabile1=is_feasible(mutated_offspring1, graph)
            new_population.append(feasabile1)
            feasabile2=is_feasible(mutated_offspring2, graph)
            new_population.append(feasabile2)

            attempts += 1

        if len(new_population) < population_size:
            print(f"Generation {generation}: Only generated {len(new_population)} out of {population_size} new solutions. Stopping early.")
            break

        population = new_population

        for chromosome in population:
            chromosome_fitness = fitness(chromosome)
            if chromosome_fitness < best_fitness_value:
                best_fitness_value = chromosome_fitness
                best_chromosome = chromosome

        print(f"Generation {generation}: Best Fitness: {best_fitness_value}")

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