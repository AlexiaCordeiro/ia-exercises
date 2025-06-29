import math
from collections import defaultdict

def ler_dados_do_arquivo(nome_arquivo):
    dados = []
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            # Pular linhas de comentário
            for linha in arquivo:
                if linha.strip().startswith('dados = ['):
                    break
            
            # Ler os dados
            for linha in arquivo:
                linha = linha.strip()
                if linha.startswith(']'):  # Fim dos dados
                    break
                if not linha or linha.startswith('#'):
                    continue
                
                # Limpar a linha e extrair valores
                linha = linha.strip().rstrip(',')
                valores = [v.strip().strip("'\" ") for v in linha.split(',')]
                if len(valores) == 5:
                    dados.append(valores)
    
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
    
    return dados

def train_naive_bayes(data):
    class_counts = defaultdict(int)
    feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    for instance in data:
        label = instance[-1]
        class_counts[label] += 1
        
        for i, feature in enumerate(instance[:-1]):
            feature_counts[i][feature][label] += 1
    
    total_instances = len(data)
    prior_probabilities = {cls: count / total_instances for cls, count in class_counts.items()}
    
    conditional_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    for feature_idx in feature_counts:
        for feature_value in feature_counts[feature_idx]:
            for label in class_counts:
                count = feature_counts[feature_idx][feature_value].get(label, 0)
                conditional_probabilities[feature_idx][feature_value][label] = (count + 1) / (class_counts[label] + len(feature_counts[feature_idx]))
    
    return prior_probabilities, conditional_probabilities, list(class_counts.keys())

def predict(instance, prior_probabilities, conditional_probabilities, classes):
    predictions = {}
    
    for cls in classes:
        predictions[cls] = math.log(prior_probabilities[cls])
        
        for i, feature_value in enumerate(instance):
            if feature_value in conditional_probabilities[i]:
                predictions[cls] += math.log(conditional_probabilities[i][feature_value][cls])
            else:
                predictions[cls] += math.log(1e-10)
    
    # Converter de volta para probabilidades
    max_log = max(predictions.values())
    for cls in predictions:
        predictions[cls] = math.exp(predictions[cls] - max_log)
    
    # Normalizar
    total = sum(predictions.values())
    for cls in predictions:
        predictions[cls] /= total
    
    return predictions

def main():
    nome_arquivo = 'Base_dados_golfe.txt'
    
    dados = ler_dados_do_arquivo(nome_arquivo)
    print(f"Dados lidos: {len(dados)} registros")
    for i, linha in enumerate(dados, 1):
        print(f"{i}: {linha}")
    
    if not dados:
        print("Nenhum dado foi lido. Verifique o arquivo.")
        return
    
    prior_probs, cond_probs, classes = train_naive_bayes(dados)
    
    test_cases = [
        ['Ensolarado', 'Quente', 'Alta', 'Fraco'],
        ['Chuvoso', 'Frio', 'Normal', 'Forte'],
        ['Nublado', 'Ameno', 'Normal', 'Fraco']
    ]
    
    for case in test_cases:

        print("\nTestando:\n")
        print("\n[Tempo, Temperatura, Umidade, Vento]")
        print(case)
        probs = predict(case, prior_probs, cond_probs, classes)
        for cls, prob in probs.items():
            print(f"{cls}: {prob*100:.2f}%")
        decisao = max(probs.items(), key=lambda x: x[1])[0]
        print(f"Decisão: {'Jogar' if decisao == 'Sim' else 'Não jogar'}")

if __name__ == "__main__":
    main()
