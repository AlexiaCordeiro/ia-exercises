from typing import List

class Particula:
    def __init__(
        self,
        posicao_atual: List[float],
        velocidade: List[float],
        pbest: List[float]
    ):
        self.posicao_atual: List[float] = posicao_atual
        self.velocidade: List[float] = velocidade
        self.pbest: List[float] = pbest