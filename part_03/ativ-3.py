from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import cm

# Optimization settings for slower convergence
POPULATION_SIZE = 400
MAX_GENERATIONS = 150
MUTATION_RATE = 0.15
CROSSOVER_RATE = 0.8
ELITISM_SIZE = 40  # Number of best individuals to keep
VIDEO_FILENAME = 'schaffer_evolution.mp4'
DOMAIN_LIMIT = 10
FPS = 15
DPI = 150

# Schaffer's F6 function
def schaffer_f6(x, y):
    term = x**2 + y**2
    numerator = (np.sin(np.sqrt(term))**2 - 0.5)
    denominator = (1 + 0.001 * term)**2
    return 0.5 - numerator / denominator

class GeneticAlgorithm:
    def __init__(self, population_size, domain_limit):
        self.population_size = population_size
        self.domain_limit = domain_limit
        # Initialize population with a normal distribution centered at (0,0)
        self.population = np.random.normal(loc=0.0, scale=3.0, size=(self.population_size, 2))
        self.population = np.clip(self.population, -self.domain_limit, self.domain_limit)

        self.best_fitness_history = []
        self.average_fitness_history = []
        self.worst_fitness_history = []
        self.best_individual_history = []
        self.population_history = []

    def calculate_fitness(self):
        return np.array([schaffer_f6(ind[0], ind[1]) for ind in self.population])

    def selection(self, fitness):
        selected = []
        for _ in range(self.population_size - ELITISM_SIZE):
            candidates_indices = np.random.choice(range(self.population_size), size=3, replace=False)
            # Original selection: always select the best from the candidates
            winner_index = candidates_indices[np.argmax(fitness[candidates_indices])]
            selected.append(self.population[winner_index])
        return np.array(selected)

    def crossover(self, selected):
        children = []
        for i in range(0, len(selected), 2):
            if i + 1 >= len(selected):
                children.append(selected[i])
                break
            if np.random.random() < CROSSOVER_RATE:
                alpha = np.random.random()
                child1 = alpha * selected[i] + (1 - alpha) * selected[i + 1]
                child2 = alpha * selected[i + 1] + (1 - alpha) * selected[i]
                children.extend([child1, child2])
            else:
                children.extend([selected[i], selected[i + 1]])
        return np.array(children)

    def mutation(self, children):
        for i in range(len(children)):
            if np.random.random() < MUTATION_RATE:
                mutation = np.random.normal(0, 1, size=2)
                children[i] += mutation
                children[i] = np.clip(children[i], -self.domain_limit, self.domain_limit)
        return children

    def elitism(self, fitness, children):
        if ELITISM_SIZE > 0:
            elite_indices = np.argsort(fitness)[-ELITISM_SIZE:]
            elite = self.population[elite_indices]
            self.population = np.vstack([children[:self.population_size - ELITISM_SIZE], elite])
        else:
            self.population = children[:self.population_size]

    def record_generation(self, fitness):
        self.best_fitness_history.append(np.max(fitness))
        self.average_fitness_history.append(np.mean(fitness))
        self.worst_fitness_history.append(np.min(fitness))
        best_index = np.argmax(fitness)
        self.best_individual_history.append(self.population[best_index].copy())
        self.population_history.append(self.population.copy())

    def run(self, generations):
        print("Optimizing...")
        for gen in tqdm(range(generations)):
            fitness = self.calculate_fitness()
            self.record_generation(fitness)
            selected = self.selection(fitness)
            children = self.crossover(selected)
            children = self.mutation(children)
            self.elitism(fitness, children)
            if self.worst_fitness_history[-1] >= self.average_fitness_history[-1]:
                print(f"Worst fitness reached average fitness at generation {gen+1}")
                break  # Stop if worst fitness reaches or exceeds average fitness

def setup_plots():
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_population, ax_fitness = axes

    # Population plot settings
    ax_population.set_xlim(-DOMAIN_LIMIT, DOMAIN_LIMIT)
    ax_population.set_ylim(-DOMAIN_LIMIT, DOMAIN_LIMIT)
    ax_population.set_xlabel('X')
    ax_population.set_ylabel('Y')
    ax_population.set_title('Population Evolution')
    ax_population.grid(True, alpha=0.3)

    # Fitness plot settings
    ax_fitness.set_xlim(0, MAX_GENERATIONS)
    ax_fitness.set_ylim(0, 1.1)
    ax_fitness.set_xlabel('Generation')
    ax_fitness.set_ylabel('Fitness')
    ax_fitness.set_title('Fitness Evolution')
    ax_fitness.grid(True, alpha=0.3)

    return fig, ax_population, ax_fitness

def create_video(ga):
    print("\nPreparing visualization...")
    fig, ax_population, ax_fitness = setup_plots()

    # Prepare contour plot data
    x = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, 100)
    y = np.linspace(-DOMAIN_LIMIT, DOMAIN_LIMIT, 100)
    X, Y = np.meshgrid(x, y)
    Z = schaffer_f6(X, Y)
    levels = np.linspace(0, 1, 20)

    # Set up video writer
    writer = FFMpegWriter(fps=FPS, metadata=dict(title='GA Evolution - Schaffer F6'))

    print("Creating video (this may take a few minutes)...")
    with writer.saving(fig, VIDEO_FILENAME, dpi=DPI):
        for gen in tqdm(range(len(ga.population_history))):
            # Update population plot
            ax_population.clear()
            ax_population.contourf(X, Y, Z, levels=levels, cmap=cm.viridis, alpha=0.6)
            population = ga.population_history[gen]
            ax_population.scatter(population[:, 0], population[:, 1], c='red', s=40,
                                 edgecolors='black', alpha=0.8, label='Individuals')
            best = ga.best_individual_history[gen]
            ax_population.scatter(best[0], best[1], c='gold', s=150, marker='*',
                                 edgecolors='black', label='Best Individual')
            ax_population.set_xlim(-DOMAIN_LIMIT, DOMAIN_LIMIT)
            ax_population.set_ylim(-DOMAIN_LIMIT, DOMAIN_LIMIT)
            ax_population.set_title(f'Population - Generation {gen+1}')
            ax_population.legend(loc='upper right')
            ax_population.grid(True, alpha=0.3)

            # Update fitness plot
            ax_fitness.clear()
            ax_fitness.plot(ga.best_fitness_history[:gen+1], 'g-', label='Best Fitness')
            ax_fitness.plot(ga.average_fitness_history[:gen+1], 'b-', label='Average Fitness')
            ax_fitness.plot(ga.worst_fitness_history[:gen+1], 'r-', label='Worst Fitness')
            ax_fitness.set_xlim(0, MAX_GENERATIONS)
            ax_fitness.set_ylim(0, 1.1)
            ax_fitness.set_title('Fitness Metrics Evolution')
            ax_fitness.legend()
            ax_fitness.grid(True, alpha=0.3)

            plt.tight_layout()
            writer.grab_frame()

    print(f"\nVideo saved as '{VIDEO_FILENAME}'")

if __name__ == "__main__":
    ga = GeneticAlgorithm(population_size=POPULATION_SIZE, domain_limit=DOMAIN_LIMIT)
    ga.run(MAX_GENERATIONS)
    create_video(ga)


