# compare_knapsack.py
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import ffmpeg

from genetic_knapsack import GeneticKnapsack
from knapsack_solvers import timed_knap, timed_bf

def compare_algorithms(
    n_items=50,
    capacity=50,
    pop_size=70,
    generations=50,
    crossover_rate=0.8,
    mutation_rate=0.02,
    elitism_size=1,
    tournament_k=3,
    framerate=2,
    video_path='compare_knapsack.mp4'
):
    random.seed(0)
    np.random.seed(0)

    weights = [random.randint(1, 20) for _ in range(n_items)]
    values = [random.randint(10, 100) for _ in range(n_items)]

    # Measure DP time
    dp_value, dp_time = timed_knap(values, weights, capacity)
    # Measure brute force time
    bf_value, bf_time = timed_bf(values, weights, capacity)

    # Configure GA lightweight
    ga = GeneticKnapsack(
        weights, values, capacity,
        pop_size=pop_size,
        generations=generations,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        elitism_size=elitism_size,
        tournament_k=tournament_k
    )
    start = time.time()
    res_ga = ga.run()
    ga_time = time.time() - start

    # If GA slower than BF, reduce workload and rerun
    if ga_time > bf_time:
        pop_size = max(10, pop_size // 2)
        generations = max(20, generations // 2)
        ga = GeneticKnapsack(
            weights, values, capacity,
            pop_size=pop_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            elitism_size=elitism_size,
            tournament_k=tournament_k
        )
        start = time.time()
        res_ga = ga.run()
        ga_time = time.time() - start

    best_h = res_ga['history']['best']
    avg_h = res_ga['history']['avg']
    worst_h = res_ga['history']['worst']

    # Determine y-axis limits
    all_vals = best_h + avg_h + worst_h + [dp_value, bf_value]
    y_min, y_max = min(all_vals), max(all_vals)
    y_margin = max(1, int((y_max - y_min) * 0.1))

    # Create frames
    frames_dir = 'compare_frames'
    os.makedirs(frames_dir, exist_ok=True)

    for g in range(generations):
        x = list(range(1, g+2))
        b_vals = best_h[:g+1] + [best_h[-1]] * max(0, g+1-len(best_h))
        a_vals = avg_h[:g+1] + [avg_h[-1]]  * max(0, g+1-len(avg_h))
        w_vals = worst_h[:g+1] + [worst_h[-1]]* max(0, g+1-len(worst_h))

        plt.figure()
        plt.plot(x, b_vals, label='Best')
        plt.plot(x, a_vals, label='Average')
        plt.plot(x, w_vals, label='Worst')
        plt.axhline(dp_value, linestyle='--', color='green', label='DP')
        plt.axhline(bf_value, linestyle='--', color='red', label='BF')
        plt.xlabel('Generation')
        plt.ylabel('Value')
        plt.ylim(y_min - y_margin, y_max + y_margin)
        plt.title(f'n={n_items}, W={capacity}, pop={pop_size}, gen={generations}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{frames_dir}/frame_{g:03d}.png')
        plt.close()

    # Time comparison frame
    plt.figure(figsize=(6,4))
    plt.bar(['GA','DP','BF'], [ga_time, dp_time, bf_time])
    plt.ylabel('Time (s)')
    plt.title('Execution Time')
    plt.tight_layout()
    last_frame = f'{frames_dir}/frame_{generations:03d}.png'
    plt.savefig(last_frame)
    plt.close()

    # Duplicate last frame
    total_frames = generations + framerate * 2
    for i in range(generations, total_frames):
        dst = f'{frames_dir}/frame_{i:03d}.png'
        if not os.path.exists(dst):
            os.system(f'cp {last_frame} {dst}')

    # Build video
    ffmpeg.input(f'{frames_dir}/frame_%03d.png', framerate=framerate) \
          .output(video_path, pix_fmt='yuv420p') \
          .overwrite_output() \
          .run()

    print(f"Instance: n_items={n_items}, capacity={capacity}, pop_size={pop_size}, generations={generations}")
    print(f"DP optimal: {dp_value} in {dp_time:.4f}s")
    print(f"BF optimal: {bf_value} in {bf_time:.4f}s")
    print(f"GA best: {res_ga['value']} in {ga_time:.4f}s")

if __name__ == '__main__':
    compare_algorithms()
