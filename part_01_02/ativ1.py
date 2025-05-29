import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FFMpegWriter, FuncAnimation

def alpine02(x):
    return np.prod(np.sqrt(x) * np.sin(x))

def clonalg_optimization_with_history(objective_func, bounds, pop_size=50, max_iter=100, 
                                    beta=1, clone_factor=0.1, mutation_rate=0.2):
    n_dim = len(bounds)
    history = {'best_fitness': [], 'best_solutions': [], 'all_populations': [], 'all_fitness': []}
    
    # Inicialização
    population = np.zeros((pop_size, n_dim))
    for i in range(n_dim):
        min_val, max_val = bounds[i]
        population[:, i] = np.random.uniform(min_val, max_val, pop_size)
    
    # Avaliação inicial
    fitness = np.array([objective_func(ind) for ind in population])
    best_idx = np.argmax(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    
    # Armazena histórico
    history['best_fitness'].append(best_fitness)
    history['best_solutions'].append(best_solution)
    history['all_populations'].append(population.copy())
    history['all_fitness'].append(fitness.copy())
    
    for iteration in range(max_iter):
        # Seleção
        n_clones = int(pop_size * clone_factor)
        sorted_indices = np.argsort(fitness)[::-1]
        selected = population[sorted_indices[:n_clones]]
        
        # Clonagem
        clones = np.zeros((0, n_dim))
        for i, individual in enumerate(selected):
            n = int(beta * pop_size / (i+1))
            clones = np.vstack([clones, np.tile(individual, (n, 1))])
        
        # Mutação
        mutated_clones = clones.copy()
        for i in range(len(clones)):
            clone_fitness = objective_func(clones[i])
            adaptive_rate = mutation_rate * (1 - clone_fitness/best_fitness)
            
            for d in range(n_dim):
                if np.random.rand() < adaptive_rate:
                    min_val, max_val = bounds[d]
                    mutated_clones[i, d] += np.random.normal(0, (max_val-min_val)/10)
                    mutated_clones[i, d] = np.clip(mutated_clones[i, d], min_val, max_val)
        
        clones_fitness = np.array([objective_func(ind) for ind in mutated_clones])
        
        # Seleção dos melhores
        combined_pop = np.vstack([population, mutated_clones])
        combined_fitness = np.concatenate([fitness, clones_fitness])
        best_indices = np.argsort(combined_fitness)[::-1][:pop_size]
        population = combined_pop[best_indices]
        fitness = combined_fitness[best_indices]
        
        # Substituição para diversidade
        n_replace = int(pop_size * 0.1)
        for i in range(n_replace):
            idx = pop_size - 1 - i
            for d in range(n_dim):
                min_val, max_val = bounds[d]
                population[idx, d] = np.random.uniform(min_val, max_val)
            fitness[idx] = objective_func(population[idx])
        
        # Atualização da melhor solução
        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_solution = population[current_best_idx].copy()
            best_fitness = fitness[current_best_idx]
        
        # Armazena histórico
        history['best_fitness'].append(best_fitness)
        history['best_solutions'].append(best_solution.copy())
        history['all_populations'].append(population.copy())
        history['all_fitness'].append(fitness.copy())
        
        if iteration % 10 == 0:
            print(f"Iteração {iteration}: Melhor fitness = {best_fitness:.4f}")
    
    return best_solution, best_fitness, history

def create_animation(history, bounds, global_max, optimal_x, filename='clonalg_evolution.mp4'):
    """Cria uma animação da evolução do algoritmo"""
    fig = plt.figure(figsize=(18, 8))
    
    # Configuração dos subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Prepara dados para o gráfico 3D
    x = np.linspace(bounds[0][0], bounds[0][1], 50)
    y = np.linspace(bounds[0][0], bounds[0][1], 50)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i,j] = alpine02(np.array([X[i,j], Y[i,j]]))
    
    # Configurações iniciais dos gráficos
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
    
    # Elementos que serão atualizados
    scatter = ax1.scatter([], [], [], c='red', s=50, label='População')
    best_point = ax1.scatter([], [], [], c='green', s=100, label='Melhor Solução')
    optimal_point = ax1.scatter([optimal_x[0]], [optimal_x[1]], [global_max], 
                              c='blue', s=100, label='Ótimo Global')
    line, = ax2.plot([], [], 'b-', label='Melhor Fitness')
    ax1.legend()
    ax2.legend()
    
    # Função de inicialização
    def init():
        scatter._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
        best_point._offsets3d = (np.empty(0), np.empty(0), np.empty(0))
        line.set_data([], [])
        return scatter, best_point, line
    
    # Função de animação
    def update(frame):
        # Atualiza gráfico 3D
        population = history['all_populations'][frame]
        fitness = history['all_fitness'][frame]
        best_sol = history['best_solutions'][frame]
        
        # Mostra apenas uma amostra da população para não sobrecarregar
        sample_size = min(30, len(population))
        indices = np.random.choice(len(population), sample_size, replace=False)
        
        x_vals = population[indices, 0]
        y_vals = population[indices, 1]
        z_vals = fitness[indices]
        
        scatter._offsets3d = (x_vals, y_vals, z_vals)
        best_point._offsets3d = ([best_sol[0]], [best_sol[1]], [history['best_fitness'][frame]])
        
        # Atualiza gráfico de convergência
        x_data = np.arange(frame+1)
        y_data = history['best_fitness'][:frame+1]
        line.set_data(x_data, y_data)
        
        # Atualiza título
        fig.suptitle(f'Iteração {frame}', fontsize=16)
        
        return scatter, best_point, line
    
    # Cria a animação
    anim = FuncAnimation(fig, update, frames=len(history['best_fitness']),
                        init_func=init, blit=False, interval=200)
    
    # Salva o vídeo (requer ffmpeg instalado)
    writer = FFMpegWriter(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    anim.save(filename, writer=writer)
    print(f"Vídeo salvo como {filename}")
    
    plt.close()

# Parâmetros do problema
n_dim = 2
bounds = [(0, 10)] * n_dim
global_max = 2.808**n_dim
optimal_x = [7.917] * n_dim

# Executa o Clonalg com histórico completo
print("Executando o algoritmo Clonalg com coleta de histórico...")
best_solution, best_fitness, history = clonalg_optimization_with_history(
    alpine02, 
    bounds, 
    pop_size=50, 
    max_iter=100,  # Reduzido para tornar a animação mais rápida
    beta=2,
    clone_factor=0.3,
    mutation_rate=0.3
)

# Resultados
print("\n=== Resultados ===")
print(f"Melhor solução encontrada: {best_solution}")
print(f"Valor da função no ponto: {best_fitness:.4f}")
print(f"Ótimo global conhecido: {global_max:.4f}")
print(f"Ponto ótimo conhecido: {optimal_x}")

# Cria a animação
print("\nCriando animação... (isso pode levar alguns minutos)")
create_animation(history, bounds, global_max, optimal_x)
