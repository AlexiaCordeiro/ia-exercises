import os
import matplotlib.pyplot as plt
import ffmpeg
from genetic_knapsack import GeneticKnapsack
from knapsack_solvers import knap_dp_bottom_up

def genetic_knapsack_fitness(
    w, v, capacity,
    pop_size=50,
    generations=100,
    crossover_rate=0.8,
    mutation_rate=0.02,
    elitism_size=1,
    tournament_k=3,
    video_path='fitness_evolution.mp4',
    framerate=2
):
    dp_opt = knap_dp_bottom_up(v, w, capacity)
    ga = GeneticKnapsack(
        w, v, capacity,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=elitism_size,
        tournament_k=tournament_k
    )
    res = ga.run()
    history = res['history']
    total_gens = generations
    os.makedirs('frames', exist_ok=True)

    best_h = history['best']
    avg_h = history['avg']
    worst_h = history['worst']
    last_best = best_h[-1]
    last_avg = avg_h[-1]
    last_worst = worst_h[-1]

    all_vals = best_h + avg_h + worst_h + [dp_opt]
    y_min = min(all_vals)
    y_max = max(all_vals)
    y_margin = max(1, int((y_max - y_min) * 0.1))

    for g in range(total_gens):
        b_vals = best_h[:g+1] if g < len(best_h) else best_h + [last_best]*(g+1-len(best_h))
        a_vals = avg_h[:g+1] if g < len(avg_h) else avg_h + [last_avg]*(g+1-len(avg_h))
        w_vals = worst_h[:g+1] if g < len(worst_h) else worst_h + [last_worst]*(g+1-len(worst_h))
        x = list(range(1, g+2))

        plt.figure()
        plt.plot(x, b_vals, label='Best')
        plt.plot(x, a_vals, label='Average')
        plt.plot(x, w_vals, label='Worst')
        plt.axhline(dp_opt, linestyle='--', color='black', label='DP')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.title(f'GA vs DP Opt (Gen {g+1})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'frames/frame_{g:03d}.png')
        plt.close()

    ffmpeg.input('frames/frame_%03d.png', framerate=framerate).output(video_path, pix_fmt='yuv420p').overwrite_output().run()
    return res

if __name__ == '__main__':
    import random, numpy as np
    random.seed(0)
    np.random.seed(0)
    w = [random.randint(1,20) for _ in range(30)]
    v = [random.randint(10,100) for _ in range(30)]
    result = genetic_knapsack_fitness(w, v, 50)
    print(f"Best weight: {result['weight']}")
    print(f"Best value: {result['value']}")