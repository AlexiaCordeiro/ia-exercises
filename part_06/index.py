from grafico import GraficoFuzzy
from gorjeta_fuzzy import GorjetaFuzzy
def main():
    fuzzy = GorjetaFuzzy()
    GraficoFuzzy.gerar_imagem(fuzzy)
    menor_gorjeta = fuzzy.min_gorjeta()
    maior_gorjeta = fuzzy.max_gorjeta()
    print(f"Menor valor possível da gorjeta: {menor_gorjeta}%")
    print(f"Maior valor possível da gorjeta: {maior_gorjeta}%")
    print(f"O maior valor é exatamente 20%? {'Sim' if maior_gorjeta == 20 else 'Não'}")

if __name__ == "__main__":
    main()