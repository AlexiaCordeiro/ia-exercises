import numpy as np
import matplotlib.pyplot as plt
import random

num_ants = 20 #maior, encontra mais facilmente
num_iterations = 200
alpha = 1.0 #peso do feromonio da decisão - maior, chega ao resultado mais rapidamente
beta = 3.0 #peso da informação heurística, que é o inverso da distância - quedas mais bruscas, maior a heuristica
rho = 0.1 #taxa de evaporação do feromômio - maior taxa, maior tempo para chegar a convergencia
Q = 1.0 
convergence_threshold = 100  #considera convergência após 100 iterações sem melhoria

def read_distance_matrix(file_path):
    with open(file_path, 'r') as file:
        distances = [[float(x) for x in line.strip().split(',')] for line in file]
    return np.array(distances)

distance_matrix = read_distance_matrix('distancia_matrix.csv')
num_cities = distance_matrix.shape[0] #considerando que cada linha é uma cidade, então a quantidade de cidades é a quantidade de linhas

pheromones = np.ones((num_cities, num_cities)) * 0.1

def calculate_probabilities(current_city, unvisited, pheromones, distances, alpha, beta):
    probabilities = []
    total = 0.0
    for city in unvisited:
        pheromone = pheromones[current_city, city] ** alpha #o valor do feromonio no caminho entre as cidades
        heuristic = (1.0 / distances[current_city, city]) ** beta #informação da heuristica que é o inverso da distância
        probabilities.append((city, pheromone * heuristic)) #armazena a cidade e o valor não normalizado
        total += pheromone * heuristic #acumula para a soma total - todo o caminho
    #normalizando - pra poder usar a roleta
    if total > 0:
        probabilities = [(city, prob/total) for city, prob in probabilities] #divide o valor pelo total pra obter a probabilidade
    else:
        probabilities = [(city, 1.0/len(unvisited)) for city in unvisited] #distribuição uniforme
    return probabilities

def select_next_city(probabilities):#usa o método da roleta
    r = random.random()
    cumulative = 0.0
    for city, prob in probabilities:
        cumulative += prob
        if r <= cumulative:
            return city
    return probabilities[-1][0] #pra evitar erros numéricos - garante que sempre retorne uma cidade

best_distance = float('inf') #inicializa como infinito, pq aí a primeira solução será sempre melhor que o valor inicial
best_path = []
convergence_counter = 0 #iterações sem melhoria
iteration_data = [] #dados pro gráfico
convergence_iteration = -1 #negativo para informar que ainda não houve convergência

#CONSTRUINDO O CAMINHO

for iteration in range(num_iterations):
    all_paths = []
    all_distances = []
    
    for ant in range(num_ants):
        current_city = random.randint(0, num_cities - 1) #cidade aleatória para cada formiga
        path = [current_city] 
        unvisited = set(range(num_cities)) - {current_city} #remove a cidade atual o set de não visitados
        distance = 0.0
        
        while unvisited:
            probabilities = calculate_probabilities(current_city, list(unvisited), pheromones, distance_matrix, alpha, beta)
            next_city = select_next_city(probabilities)
            path.append(next_city)
            distance += distance_matrix[current_city, next_city]
            unvisited.remove(next_city)
            current_city = next_city
        
        distance += distance_matrix[path[-1], path[0]] #calcula a distancia da volta (cidade final até cidade inicial)
        path.append(path[0])
        
        all_paths.append(path)
        all_distances.append(distance)
    
    pheromones *= (1.0 - rho) #calculo da evaporação dos feromonios
    
    for path, distance in zip(all_paths, all_distances):
        delta_pheromone = Q / distance #caminhos mais curtos = mais feromonios 
        for i in range(len(path) - 1):
            city_from = path[i]
            city_to = path[i+1]
            pheromones[city_from, city_to] += delta_pheromone #adiciona feromonios
            pheromones[city_to, city_from] += delta_pheromone #fazendo isso para manter a matriz simétrica
    
    iteration_best_distance = min(all_distances)
    iteration_best_path = all_paths[all_distances.index(iteration_best_distance)]
    
    if iteration_best_distance < best_distance:
        best_distance = iteration_best_distance
        best_path = iteration_best_path
        convergence_counter = 0 
    else:
        convergence_counter += 1
    
    iteration_data.append(best_distance)
    
    if convergence_counter >= convergence_threshold and convergence_iteration == -1:
        convergence_iteration = iteration - convergence_threshold
        print(f"\nSolução convergiu na iteração {convergence_iteration} (permaneceu estável por {convergence_threshold} iterações)")
        print(f"Melhor distância encontrada: {best_distance:.2f}")

print("\nResultado final:")
print(f"Melhor distância: {best_distance:.2f}")
print(f"Melhor caminho: {best_path}")

plt.figure(figsize=(12, 6))
plt.plot(iteration_data, 'b-', label='Melhor distância')
if convergence_iteration != -1:
    plt.axvline(x=convergence_iteration, color='r', linestyle='--', 
                label=f'Convergência (iteração {convergence_iteration})')
plt.xlabel('Iteração')
plt.ylabel('Distância')
plt.title('Colônia de Formigas')
plt.legend()
plt.grid(True)
plt.show()

print("\nDetalhes do melhor caminho:")
total = 0
for i in range(len(best_path) - 1):
    dist = distance_matrix[best_path[i], best_path[i+1]]
    total += dist
    print(f"{best_path[i]} → {best_path[i+1]}: {dist:.2f} (subtotal: {total:.2f})")
