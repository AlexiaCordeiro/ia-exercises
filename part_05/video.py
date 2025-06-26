import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation

def rastrigin(X, Y):
    return 20 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))

def rastrigin_fitness(pos):
    X = np.asarray(pos)[..., 0]
    Y = np.asarray(pos)[..., 1]
    return 20 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))

def gerar_video(historico_posicoes, max_iters, output_file='pso_rastrigin.mp4'):
    x = np.linspace(-5.12, 5.12, 200)
    y = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(X, Y)

    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1], wspace=0.04)
    ax = fig.add_subplot(gs[0])
    contourf = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    contour = ax.contour(X, Y, Z, levels=50, colors='white', linewidths=0.5)
    particles_plot, = ax.plot([], [], marker='o', color='red', linestyle='', markersize=5, label="Partículas")
    best_plot, = ax.plot([], [], marker='*', color='gold', markersize=16, linestyle='', label="Melhor indivíduo")

    ax.set_xlim(-5.12, 5.12)
    ax.set_ylim(-5.12, 5.12)
    ax.set_aspect('equal')
    ax.set_title("Otimização PSO - Função Rastrigin")

    cax = fig.add_subplot(gs[1])
    plt.colorbar(contourf, cax=cax)

    ax.legend(loc="upper right")

    def atualizar(frame):
        posicoes = np.array(historico_posicoes[frame])  
        if posicoes.ndim == 1:
            if posicoes.size == 0:
                particles_plot.set_data([], [])
                best_plot.set_data([], [])
            else:
                particles_plot.set_data([posicoes[0]], [posicoes[1]])
                best_plot.set_data([posicoes[0]], [posicoes[1]])
        else:
            particles_plot.set_data(posicoes[:, 0], posicoes[:, 1])
            fitness_vals = rastrigin_fitness(posicoes)
            idx_best = np.argmin(fitness_vals)
            best_pos = posicoes[idx_best]
            best_plot.set_data([best_pos[0]], [best_pos[1]])
        ax.set_xlabel(f"Iteração {frame+1}/{max_iters}")
        return particles_plot, best_plot

    ani = animation.FuncAnimation(fig, atualizar, frames=max_iters, interval=50, blit=True)
    ani.save(output_file, writer='ffmpeg', fps=20)
    plt.close(fig)
