import math
import random

from typing import List, Tuple
from particula import Particula

class EnxameParticulas:
    def __init__(
        self,
        intervalo_valido: Tuple[float, float],
        numero_particulas: int,
        numero_variaveis: int,
        c1: float,
        c2: float
    ):
        self.intervalo_valido = intervalo_valido
        self.numero_particulas = numero_particulas
        self.numero_variaveis = numero_variaveis
        self.c1 = c1
        self.c2 = c2

    def gerar_populacao(self) -> List[Particula]:
        limite_inferior: float = self.intervalo_valido[0]
        limite_superior: float = self.intervalo_valido[1]
        populacao_particulas: List[Particula] = []
        for _ in range(self.numero_particulas):
            posicao: List[float] = [
                random.uniform(limite_inferior, limite_superior)
                for _ in range(self.numero_variaveis)
            ]
            amplitude = 0.01 * abs(limite_superior - limite_inferior)
            velocidade: List[float] = [
                random.uniform(-amplitude, amplitude)
                for _ in range(self.numero_variaveis)
            ]
            particula = Particula(posicao, velocidade, posicao[:])
            populacao_particulas.append(particula)
        return populacao_particulas
    
    def calcular_fitness(self, posicao: List[float]) -> float:
        somatorio: float = 0.0
        for xi in posicao:
            somatorio += xi**2 - 10 * math.cos(2 * math.pi * xi)
        return 10 * self.numero_variaveis + somatorio

    def atualizar_pbest(self, particulas: List[Particula]) -> None:
        for particula in particulas:
            fitness_atual = self.calcular_fitness(particula.posicao_atual)
            fitness_pbest = self.calcular_fitness(particula.pbest)
            if fitness_atual < fitness_pbest:
                particula.pbest = particula.posicao_atual[:]

    def obter_gbest(self, particulas: List[Particula]) -> List[float]:
        melhor_fitness = float('inf')
        gbest: List[float] = []
        for particula in particulas:
            fitness_pbest = self.calcular_fitness(particula.pbest)
            if fitness_pbest < melhor_fitness:
                melhor_fitness = fitness_pbest
                gbest = particula.pbest[:] 
        return gbest
    
    def atualizar_velocidade(
    self,
    particulas: List[Particula],
    gbest_pos: List[float],
    k: int,
    k_max: int,
    w: float = 0.7
    ) -> None:
        for particula in particulas:
            for i in range(self.numero_variaveis):
                r1: float = random.random()
                r2: float = random.random()
                velocidade_atual = particula.velocidade[i]
                posicao_atual = particula.posicao_atual[i]
                pbest = particula.pbest[i]
                gbest = gbest_pos[i]
                w = 0.9 - k * (0.5 / k_max)
                nova_velocidade = (
                    w * velocidade_atual
                    + self.c1 * r1 * (pbest - posicao_atual)
                    + self.c2 * r2 * (gbest - posicao_atual)
                )
                particula.velocidade[i] = nova_velocidade

    def atualizar_posicao(self, particulas: List[Particula]) -> None:
        for particula in particulas:
            for i in range(self.numero_variaveis):
                nova_posicao = particula.posicao_atual[i] + particula.velocidade[i]
                limite_inferior: float = self.intervalo_valido[0]
                limite_superior: float = self.intervalo_valido[1]
                # Clamp na posição
                if nova_posicao > limite_superior:
                    nova_posicao = limite_superior
                if nova_posicao < limite_inferior:
                    nova_posicao = limite_inferior
                particula.posicao_atual[i] = nova_posicao
