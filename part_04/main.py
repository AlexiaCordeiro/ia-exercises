import numpy as np
import matplotlib.pyplot as plt
import random

# Parâmetros do algoritmo
num_ants = 20
num_iterations = 150
alpha = 1.0
beta = 3.0
rho = 0.1
Q = 1.0
convergence_threshold = 100  # Considera convergência após 100 iterações sem melhoria

# Função para ler a matriz de distâncias do arquivo CSV
def read_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        distances = [[float(x) for x in line.strip().split(',')] for line in file]
    return np.array(distances)

# Carrega a matriz de distâncias
distance_matrix = read_distance_matrix('distancia_matrix.csv')
num_cities = distance_matrix.shape[0]

# Inicializa feromônios
pheromones = np.ones((num_cities, num_cities)) * 0.1

# Função para calcular probabilidades
def calculate_probabilities(current_city, unvisited, pheromones, distances, alpha, beta):
    probabilities = []
    total = 0.0
    for city in unvisited:
        pheromone = pheromones[current_city, city] ** alpha
        heuristic = (1.0 / distances[current_city, city]) ** beta
        probabilities.append((city, pheromone * heuristic))
        total += pheromone * heuristic
    if total > 0:
        probabilities = [(city, prob/total) for city, prob in probabilities]
    else:
        probabilities = [(city, 1.0/len(unvisited)) for city in unvisited]
    return probabilities

# Função para selecionar próxima cidade
def select_next_city(probabilities):
    r = random.random()
    cumulative = 0.0
    for city, prob in probabilities:
        cumulative += prob
        if r <= cumulative:
            return city
    return probabilities[-1][0]

# Algoritmo principal
best_distance = float('inf')
best_path = []
convergence_counter = 0
iteration_data = []
convergence_iteration = -1

for iteration in range(num_iterations):
    all_paths = []
    all_distances = []
    
    # Cada formiga constrói um caminho
    for ant in range(num_ants):
        current_city = random.randint(0, num_cities - 1)
        path = [current_city]
        unvisited = set(range(num_cities)) - {current_city}
        distance = 0.0
        
        while unvisited:
            probabilities = calculate_probabilities(current_city, list(unvisited), pheromones, distance_matrix, alpha, beta)
            next_city = select_next_city(probabilities)
            path.append(next_city)
            distance += distance_matrix[current_city, next_city]
            unvisited.remove(next_city)
            current_city = next_city
        
        # Completa o ciclo retornando à cidade inicial
        distance += distance_matrix[path[-1], path[0]]
        path.append(path[0])
        
        all_paths.append(path)
        all_distances.append(distance)
    
    # Evaporação de feromônios
    pheromones *= (1.0 - rho)
    
    # Depósito de feromônios
    for path, distance in zip(all_paths, all_distances):
        delta_pheromone = Q / distance
        for i in range(len(path) - 1):
            city_from = path[i]
            city_to = path[i+1]
            pheromones[city_from, city_to] += delta_pheromone
            pheromones[city_to, city_from] += delta_pheromone  # Matriz simétrica
    
    # Encontra a melhor solução desta iteração
    iteration_best_distance = min(all_distances)
    iteration_best_path = all_paths[all_distances.index(iteration_best_distance)]
    
    # Atualiza a melhor solução global
    if iteration_best_distance < best_distance:
        best_distance = iteration_best_distance
        best_path = iteration_best_path
        convergence_counter = 0  # Reseta o contador
    else:
        convergence_counter += 1
    
    # Armazena dados para o gráfico
    iteration_data.append(best_distance)
    
    # Verifica convergência
    if convergence_counter >= convergence_threshold and convergence_iteration == -1:
        convergence_iteration = iteration - convergence_threshold
        print(f"\nSolução convergiu na iteração {convergence_iteration} (permaneceu estável por {convergence_threshold} iterações)")
        print(f"Melhor distância encontrada: {best_distance:.2f}")
        # Não break para continuar plotando

# Resultados finais
print("\nResultado final:")
print(f"Melhor distância: {best_distance:.2f}")
print(f"Melhor caminho: {best_path}")

# Plotagem do gráfico de convergência
plt.figure(figsize=(12, 6))
plt.plot(iteration_data, 'b-', label='Melhor distância')
if convergence_iteration != -1:
    plt.axvline(x=convergence_iteration, color='r', linestyle='--', 
                label=f'Convergência (iteração {convergence_iteration})')
plt.xlabel('Iteração')
plt.ylabel('Distância')
plt.title('Evolução da Solução do Caixeiro Viajante com Colônia de Formigas')
plt.legend()
plt.grid(True)
plt.show()

# Detalhamento do melhor caminho
print("\nDetalhes do melhor caminho:")
total = 0
for i in range(len(best_path) - 1):
    dist = distance_matrix[best_path[i], best_path[i+1]]
    total += dist
    print(f"{best_path[i]} → {best_path[i+1]}: {dist:.2f} (subtotal: {total:.2f})")
