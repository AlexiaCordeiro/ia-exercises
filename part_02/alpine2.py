import os
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg
import random

# Alpine02 (produto) para n=2
def alpine2(pop, bounds=None):
    terms = np.sqrt(pop) * np.sin(pop)
    return np.prod(terms, axis=1)

# Parâmetros
POP_SIZE    = 100
GEN_MAX     = 50
CX_RATE     = 0.8
MUT_RATE    = 0.1
ELITE_SIZE  = 1
VAR_RANGE   = [0, 10]

# SUS Selection
def sus_selection(pop, fitness, n):
    total_fitness = np.sum(fitness - fitness.min() + 1e-6)
    step = total_fitness / n
    start = random.uniform(0, step)
    pointers = start + step * np.arange(n)
    cum = np.cumsum(fitness - fitness.min() + 1e-6)
    indices = []
    i = 0
    for p in pointers:
        while cum[i] < p:
            i += 1
        indices.append(i)
    return pop[indices]

# Preparar grade para contorno
x = np.linspace(VAR_RANGE[0], VAR_RANGE[1], 200)
y = np.linspace(VAR_RANGE[0], VAR_RANGE[1], 200)
X, Y = np.meshgrid(x, y)
Z = alpine2(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)

# Inicialização
population = np.random.uniform(VAR_RANGE[0], VAR_RANGE[1], size=(POP_SIZE, 2))
fitness_history = {'best': [], 'avg': [], 'worst': []}

# Diretório de frames
frames_dir = 'frames'
os.makedirs(frames_dir, exist_ok=True)

# Evolução e gravação de frames
for gen in range(GEN_MAX):
    fitness = alpine2(population)

    fitness_history['best'].append(np.max(fitness))
    fitness_history['avg'].append(np.mean(fitness))
    fitness_history['worst'].append(np.min(fitness))

    # Elitismo
    elite_idx = np.argsort(fitness)[-ELITE_SIZE:]
    elite = population[elite_idx]

    # Seleção
    parents = sus_selection(population, fitness, POP_SIZE - ELITE_SIZE)

    # Crossover uniforme
    children = []
    for i in range(0, len(parents), 2):
        if i+1 >= len(parents):
            break
        p1, p2 = parents[i], parents[i+1]
        mask = np.random.randint(0, 2, size=2)
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        children.extend([c1, c2])
    children = np.array(children)

    # Mutação gaussiana
    noise = np.random.normal(0, 0.5, size=children.shape)
    m_mask = np.random.rand(*children.shape) < MUT_RATE
    children = np.clip(children + noise * m_mask, VAR_RANGE[0], VAR_RANGE[1])
    # --------
    population = np.vstack([children, elite])

    plt.figure(figsize=(6,6))
    plt.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.scatter(population[:,0], population[:,1], color='red', s=20)
    plt.title(f'Geração {gen+1}')
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.tight_layout()
    plt.savefig(f'{frames_dir}/frame_{gen:03d}.png')
    plt.close()


(
    ffmpeg
    .input(f'{frames_dir}/frame_%03d.png', framerate=5)
    .output('alpine2_evolution.mp4', pix_fmt='yuv420p')
    .overwrite_output()
    .run()
)

# Exibir resultados finais
best_idx = np.argmax(alpine2(population))
best = population[best_idx]
best_val = alpine2(np.array([best]))[0]
print(f"Melhor solução encontrada: x1={best[0]:.4f}, x2={best[1]:.4f}")
print(f"Valor da função: {best_val:.4f}")
