import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter, FuncAnimation

def alpine02(x):
    return np.prod(np.sqrt(x) * np.sin(x)) #Fórmula - prod é a biblioteca do numpy para checar o produto

def clonalg_optimization_with_history(objective_func, bounds, pop_size, max_iter, 
                                    beta, clone_factor, mutation_rate):
    n_dim = 2
    history = {'best_fitness': [], 'best_solutions': [], 'all_populations': [], 'all_fitness': []}
    
    population = np.zeros((pop_size, n_dim)) #inicializando a população
    for i in range(n_dim):
        min_val, max_val = bounds[i] #bounds possui os limites. Pega cada extremo de cada dimenção
        population[:, i] = np.random.uniform(min_val, max_val, pop_size) #gera valores aleatórios entre os limites. Preenche as linhas, iterando sobre as colunas
    
    fitness = np.array([objective_func(ind) for ind in population]) #fórmula do alpine para calcular o fitness de cada indivíduo
    best_idx = np.argmax(fitness) #o index do maior valor de fitness
    best_solution = population[best_idx].copy() #o valor do maior fitness
    best_fitness = fitness[best_idx] #Pega o melhor fitness
    
    history['best_fitness'].append(best_fitness)
    history['best_solutions'].append(best_solution)
    history['all_populations'].append(population.copy())
    history['all_fitness'].append(fitness.copy())
    
    #CLONAGEM

    for iteration in range(max_iter): #passa pelas eras
        n_clones = int(pop_size * clone_factor) #calcula quantidade de clones
        sorted_indices = np.argsort(fitness)[::-1] #pega os índices dos valores do fitness ordenados em ordem descrescente 
        selected = population[sorted_indices[:n_clones]] #selecionar os melhores indivíduos da população 
        
        clones = np.zeros((0, n_dim)) #inicializa a matriz vazia
        for i, individual in enumerate(selected): #individual é o vetor
            n = int(beta * pop_size / (i+1)) # Faz o número de clones decair com a posição no ranking (beta=1, pop_size=50, i=0 )
            clones = np.vstack([clones, np.tile(individual, (n, 1))]) #tile replica o individuo n vezes na vertical e vstack empilha os novos clones em uma nova matriz
        
        #FAZ A MUTAÇÃO
        mutated_clones = clones.copy()
        for i in range(len(clones)): #linha
            clone_fitness = objective_func(clones[i]) #usa a fómula para encontrar os fitness
            adaptive_rate = mutation_rate * (1 - clone_fitness/best_fitness) #Chance de mutação. Quanto melhor o fitness, menos chances de mutação
            
            for d in range(n_dim): #mutação para cada dimensão
                if np.random.rand() < adaptive_rate:
                    min_val, max_val = bounds[d] #pega os limites de cada dimensão
                    mutated_clones[i, d] += np.random.normal(0, (max_val-min_val)/10) #adiciona ruído gaussiano para explorar o espaço de busca
                    mutated_clones[i, d] = np.clip(mutated_clones[i, d], min_val, max_val)#garante que os valores estejam dentro dos limites. Evita valores mais distantes
        
        clones_fitness = np.array([objective_func(ind) for ind in mutated_clones]) #calcula o fitness dos clones mutados
       
        #selecionar os melhores indivíduos mutados
        combined_pop = np.vstack([population, mutated_clones]) #concatena verticalmente a população original com os clones mutados
        combined_fitness = np.concatenate([fitness, clones_fitness]) #Junta os vetores de fitness da população original e dos clones
        best_indices = np.argsort(combined_fitness)[::-1][:pop_size] #ordena com os melhores indices primeiro pegando os valores até o tamanho de pop_size
        population = combined_pop[best_indices]
        fitness = combined_fitness[best_indices]
       
        #substituindo os piores indivíduos por soluções aleatórias
        n_replace = int(pop_size * 0.1) #substitui 10% da população
        for i in range(n_replace):
            idx = pop_size - 1 - i #indice dos piores indivíduos
            for d in range(n_dim): #pra cada dimensão
                min_val, max_val = bounds[d] #limites de cada dimensão
                population[idx, d] = np.random.uniform(min_val, max_val) #Pra cada dimensão do indivíduo gera um valor aleatório dentro dos limites
            fitness[idx] = objective_func(population[idx]) #calcula o fitness dos novos valores
        
        current_best_idx = np.argmax(fitness) #index do maior valor de fitness
        if fitness[current_best_idx] > best_fitness:
            best_solution = population[current_best_idx].copy() #atualiza com a nova melhor solução
            best_fitness = fitness[current_best_idx] #atualiza com o novo melhjor fitness
        
        history['best_fitness'].append(best_fitness)
        history['best_solutions'].append(best_solution.copy())
        history['all_populations'].append(population.copy())
        history['all_fitness'].append(fitness.copy())
        
    
    return best_solution, best_fitness, history

def create_animation(history, bounds, global_max, optimal_x, filename='clonalg_evolution.mp4'):
    """Cria uma animação da evolução do algoritmo"""
    fig = plt.figure(figsize=(18, 8))
    
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[0][0], bounds[0][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = alpine02(np.array([X[i,j], Y[i,j]]))
    
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.5)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('f(x)')
    ax1.set_title('Evolução da População')
    ax1.set_zlim(0, global_max*1.2)
    
    ax2.set_xlim(0, len(history['best_fitness']))
    ax2.set_ylim(0, global_max*1.2)
    ax2.set_xlabel('Iteração')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Convergência do Algoritmo')
    ax2.axhline(y=global_max, color='r', linestyle='--', label='Ótimo Global')
    ax2.grid()
    
    scatter = ax1.scatter([], [], [], c='red', s=50, label='População')
    best_point = ax1.scatter([], [], [], c='green', s=100, label='Melhor Solução')
    optimal_point = ax1.scatter([optimal_x[0]], [optimal_x[1]], [global_max], 
                              c='blue', s=100, label='Ótimo Global')
    line, = ax2.plot([], [], 'b-', label='Melhor Fitness')
    ax1.legend()
    ax2.legend()
    
    def init():
        scatter._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
        best_point._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
        line.set_data([], [])
        return scatter, best_point, line
    
    def update(frame):
        population = history['all_populations'][frame]
        fitness = history['all_fitness'][frame]
        best_sol = history['best_solutions'][frame]
        
        sample_size = min(30, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        
        x_vals = population[indices, 0]
        y_vals = population[indices, 1]
        z_vals = fitness[indices]
        
        scatter._offsets3d = (x_vals, y_vals, z_vals)
        best_point._offsets3d = ([best_sol[0]], [best_sol[1]], [history['best_fitness'][frame]])
        
        x_data = np.arange(frame+1)
        y_data = history['best_fitness'][:frame+1]
        line.set_data(x_data, y_data)
        
        fig.suptitle(f'Iteração {frame}', fontsize=16)
        
        return scatter, best_point, line
    
    anim = FuncAnimation(fig, update, frames=len(history['best_fitness']),
                        init_func=init, blit=False, interval=200)
    
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filename, writer=writer)
    print(f"Vídeo salvo como {filename}")
    
    plt.close()

n_dim = 2
bounds = [(0, 10)] * n_dim #limite de 10, pq o maximo que chegará pe em 7.917

#primeiro positivo máximo = 7,917
global_max = 2.808**n_dim
optimal_x = [7.917] * n_dim

best_solution, best_fitness, history = clonalg_optimization_with_history(
    alpine02, 
    bounds, 
    pop_size=400, 
    max_iter=100,
    beta=2, #quant de clones
    clone_factor=0.3,
    mutation_rate=0.3
)

print("\n=== Resultados ===")
print(f"Melhor solução encontrada: {best_solution}")
print(f"Valor da função no ponto: {best_fitness:.4f}")

print("\nCriando animação...")
create_animation(history, bounds, global_max, optimal_x)
