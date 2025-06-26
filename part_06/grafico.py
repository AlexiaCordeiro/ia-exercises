import numpy as np
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gorjeta_fuzzy import GorjetaFuzzy

class GraficoFuzzy:
    def gerar_imagem(fuzzy: GorjetaFuzzy, nome_arquivo='superficie_gorjeta.png'):
        qualidade_comida = np.linspace(0, 1, 50)
        qualidade_servico = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(qualidade_comida, qualidade_servico)
        Z = np.zeros_like(X)
        for i in range(50):
            for j in range(50):
                Z[i, j] = fuzzy.simular(X[i, j], Y[i, j])
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel('Qualidade da Comida')
        ax.set_ylabel('Qualidade do Serviço')
        ax.set_zlabel('Gorjeta (%)')
        plt.title('Superfície 3D para Gorjeta')
        plt.savefig(nome_arquivo)
        plt.close(fig)


