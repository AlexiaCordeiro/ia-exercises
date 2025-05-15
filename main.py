from tqdm import tqdm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import cm

# Configurações
POP_SIZE = 300  # Reduzido para melhor visualização
GEN_MAX = 100   # Reduzido para testes mais rápidos
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
ELITISM = 1
VIDEO_FILE = 'schaffer_evolution.mp4'

# Função Schaffer's F6
def schaffer_f6(x, y):
    termo = x**2 + y**2
    numerator = (np.sin(np.sqrt(termo))**2 - 0.5)
    denominator = (1 + 0.001 * termo)**2
    return 0.5 - numerator / denominator

class GeneticAlgorithm:
    def __init__(self):
        self.population = np.random.uniform(-10, 10, (POP_SIZE, 2))  # Intervalo reduzido
        self.best_fitness = []
        self.avg_fitness = []
        self.worst_fitness = []
        self.best_individual_history = []
        self.population_history = []
        
    def calculate_fitness(self):
        return np.array([schaffer_f6(ind[0], ind[1]) for ind in self.population])
    
    def run(self, generations):
        print("Otimizando...")
        for gen in tqdm(range(generations)):
            fitness = self.calculate_fitness()
            
            # Armazenar métricas
            self.best_fitness.append(np.max(fitness))
            self.avg_fitness.append(np.mean(fitness))
            self.worst_fitness.append(np.min(fitness))
            
            best_idx = np.argmax(fitness)
            self.best_individual_history.append(self.population[best_idx].copy())
            self.population_history.append(self.population.copy())
            
            # Seleção por torneio
            selected = []
            for _ in range(POP_SIZE - ELITISM):
                candidates = np.random.choice(range(POP_SIZE), size=3, replace=False)
                winner = candidates[np.argmax(fitness[candidates])]
                selected.append(self.population[winner])
            selected = np.array(selected)
            
            # Crossover
            children = []
            for i in range(0, len(selected), 2):
                if i+1 >= len(selected):
                    children.append(selected[i])
                    break
                if np.random.random() < CROSSOVER_RATE:
                    alpha = np.random.random()
                    child1 = alpha * selected[i] + (1-alpha) * selected[i+1]
                    child2 = alpha * selected[i+1] + (1-alpha) * selected[i]
                    children.extend([child1, child2])
                else:
                    children.extend([selected[i], selected[i+1]])
            children = np.array(children)
            
            # Mutação
            for i in range(len(children)):
                if np.random.random() < MUTATION_RATE:
                    children[i] += np.random.normal(0, 1, size=2)
                    children[i] = np.clip(children[i], -10, 10)
            
            # Elitismo
            elite_indices = np.argsort(fitness)[-ELITISM:]
            self.population = np.vstack([children[:POP_SIZE-ELITISM], 
                                       self.population[elite_indices]])
            
def setup_plots():
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(14, 6))
    
    # Gráfico da população e curvas de nível
    ax1 = fig.add_subplot(121)
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Evolução da População')
    ax1.grid(True, alpha=0.3)
    
    # Gráfico de linhas com as métricas
    ax2 = fig.add_subplot(122)
    ax2.set_xlim(0, GEN_MAX)
    ax2.set_ylim(0, 1.1)
    ax2.set_xlabel('Geração')
    ax2.set_ylabel('Fitness')
    ax2.set_title('Evolução do Fitness')
    ax2.grid(True, alpha=0.3)
    
    return fig, ax1, ax2

def create_video(ga):
    print("\nPreparando visualização...")
    fig, ax1, ax2 = setup_plots()
    
    # Preparar curvas de nível
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = schaffer_f6(X, Y)
    levels = np.linspace(0, 1, 20)
    
    # Configurar o escritor de vídeo
    writer = FFMpegWriter(fps=10, metadata=dict(title='Evolução do AG - Schaffer F6'))
    
    print("Criando vídeo (pode demorar alguns minutos)...")
    with writer.saving(fig, VIDEO_FILE, dpi=100):
        for gen in tqdm(range(len(ga.population_history))):
            # Gráfico da população
            ax1.clear()
            contour = ax1.contourf(X, Y, Z, levels=levels, cmap=cm.viridis, alpha=0.6)
            
            population = ga.population_history[gen]
            ax1.scatter(population[:, 0], population[:, 1], c='red', s=40, 
                       edgecolors='black', alpha=0.8, label='Indivíduos')
            
            best = ga.best_individual_history[gen]
            ax1.scatter(best[0], best[1], c='gold', s=150, marker='*', 
                       edgecolors='black', label='Melhor Indivíduo')
            
            ax1.set_xlim(-10, 10)
            ax1.set_ylim(-10, 10)
            ax1.set_title(f'População - Geração {gen+1}')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Gráfico de linhas
            ax2.clear()
            ax2.plot(ga.best_fitness[:gen+1], 'g-', label='Melhor Fitness')
            ax2.plot(ga.avg_fitness[:gen+1], 'b-', label='Fitness Médio')
            ax2.plot(ga.worst_fitness[:gen+1], 'r-', label='Pior Fitness')
            
            ax2.set_xlim(0, GEN_MAX)
            ax2.set_ylim(0, 1.1)
            ax2.set_title('Evolução das Métricas de Fitness')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            writer.grab_frame()
            
    
    print(f"\nVídeo salvo como '{VIDEO_FILE}'")

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    ga.run(GEN_MAX)
    create_video(ga)

