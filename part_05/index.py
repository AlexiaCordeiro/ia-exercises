from enxame_particulas import EnxameParticulas
from video import gerar_video

# Parâmetros do PSO
NUM_PARTICULAS = 25
NUM_VARIAVEIS = 2
C1 = 2.0
C2 = 2.0
MAX_ITERS = 200
INTERVALO_VALIDO = (-5.12, 5.12)

def main():
    pso = EnxameParticulas(INTERVALO_VALIDO, NUM_PARTICULAS, NUM_VARIAVEIS, C1, C2)
    particulas = pso.gerar_populacao()
    gbest_pos = particulas[0].pbest[:]
    gbest_val = pso.calcular_fitness(particulas[0].pbest)

    historico_posicoes = []

    for i in range(MAX_ITERS):
        pos_atual = []
        pso.atualizar_pbest(particulas)
        gbest_pos = pso.obter_gbest(particulas)
        gbest_val = pso.calcular_fitness(gbest_pos)
        for particula in particulas:
            pos_atual.append(particula.posicao_atual[:])
        historico_posicoes.append(pos_atual)
        pso.atualizar_velocidade(particulas, gbest_pos, i, MAX_ITERS)
        pso.atualizar_posicao(particulas)

    print("Melhor valor encontrado (gbest):", gbest_val)
    print("Posição correspondente (gbest_pos):", gbest_pos)

    # Chama a função para gerar o vídeo
    gerar_video(historico_posicoes, MAX_ITERS)

if __name__ == "__main__":
    main()
