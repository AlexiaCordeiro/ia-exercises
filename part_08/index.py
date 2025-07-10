import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Ler a base de dados
df = pd.read_csv('./heart.csv')

# 2. Separar features e alvo
X = df.drop('HeartDisease', axis=1)
X = pd.get_dummies(X)
y = df['HeartDisease']

# Funções para executar o experimento 30x
def evaluate_classifier(clf_class, params=None, runs=30):
    accuracies = []
    for seed in range(runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        clf = clf_class(**(params if params else {}))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(accuracies), np.std(accuracies)

# 3. Árvores de decisão
dt_mean, dt_std = evaluate_classifier(DecisionTreeClassifier, params={'max_depth': 5})

# 4. Random Forest
rf_mean, rf_std = evaluate_classifier(RandomForestClassifier)

print(f"Árvore de Decisão: média={dt_mean:.2f}, std={dt_std:.2f}")
print(f"Random Forest: média={rf_mean:.2f}, std={rf_std:.2f}")

# 5. Variação da acurácia com profundidade máxima (Random Forest)
max_features = X.shape[1]
depths = range(1, max_features + 1)
depth_accs = []

for d in depths:
    mean_acc, _ = evaluate_classifier(RandomForestClassifier, params={'max_depth': d})
    depth_accs.append(mean_acc)

plt.figure()
plt.plot(depths, depth_accs, marker='o')
plt.xlabel('max_depth')
plt.ylabel('Média da Acurácia')
plt.title('Média da Acurácia vs Profundidade da Árvore (max_depth)')
plt.grid()
plt.savefig('acuracia_profundidade.png')