import math
from collections import defaultdict

def ler_dados_do_arquivo(nome_arquivo):
    dados = []
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as arquivo:
            for linha in arquivo:
                if linha.strip().startswith('dados = ['):
                    break
            
            for linha in arquivo:
                linha = linha.strip()
                if linha.startswith(']'):
                    break
                if not linha or linha.startswith('#'):
                    continue
                
                linha = linha.strip().rstrip(',')
                line_cleaned = linha.lstrip("['").rstrip("']")
                valores = [v.strip().strip("'\" ") for v in line_cleaned.split(',')]
                
                if len(valores) == 5:
                    dados.append(valores)
    except Exception as e:
        print(f"Erro ao ler arquivo: {e}")
        
    return dados

def train_naive_bayes(data):
    class_counts = defaultdict(int)
    feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    unique_feature_values = defaultdict(set) 
    
    for instance in data:
        label = instance[-1]
        class_counts[label] += 1
        
        for i, feature in enumerate(instance[:-1]):
            feature_counts[i][feature][label] += 1
            unique_feature_values[i].add(feature) 
    
    total_instances = len(data)
    prior_probabilities = {cls: count / total_instances for cls, count in class_counts.items()}
    
    conditional_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    
    for feature_idx in feature_counts:
        for feature_value in feature_counts[feature_idx]:
            for label in class_counts:
                count = feature_counts[feature_idx][feature_value].get(label, 0)
                
                if class_counts[label] > 0:
                    conditional_probabilities[feature_idx][feature_value][label] = count / class_counts[label]
                else:
                    conditional_probabilities[feature_idx][feature_value][label] = 0.0
    
    return prior_probabilities, conditional_probabilities, list(class_counts.keys())

def predict(instance, prior_probabilities, conditional_probabilities, classes):
    predictions = {}
    
    for cls in classes:
        log_prob_sum = math.log(prior_probabilities[cls])
        
        for i, feature_value in enumerate(instance):
            prob = conditional_probabilities[i][feature_value].get(cls, 1e-10) 
            
            if prob == 0:
                prob = 1e-10 
                
            log_prob_sum += math.log(prob)
        predictions[cls] = log_prob_sum
    
    exp_predictions = {cls: math.exp(log_prob) for cls, log_prob in predictions.items()}
    
    total_exp = sum(exp_predictions.values())
    
    final_probabilities = {cls: exp_prob / total_exp for cls, exp_prob in exp_predictions.items()}
    
    return final_probabilities

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
        ['Chuvoso', 'Quente', 'Normal', 'Fraco'],
        ['Nublado', 'Ameno', 'Normal', 'Fraco']
    ]
    
    for case in test_cases:
        print("\nTestando:\n")
        print("\n[Tempo, Temperatura, Umidade, Vento]")
        print(case)
        probs = predict(case, prior_probs, cond_probs, classes)
        for cls, prob in probs.items():
            print(f"{cls}: {prob*100:.0f}%")
        
        decisao = max(probs.items(), key=lambda x: x[1])[0]
        if decisao == "Sim":
            print(f"Decisão: Jogar")
        elif decisao == "Nao":
            print(f"Decisão: Não jogar")
        else:
            print(f"Decisão: {decisao}")
        
if __name__ == "__main__":
    main()
