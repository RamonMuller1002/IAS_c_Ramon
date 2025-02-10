# Importar bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Função para criar e treinar uma Árvore de Decisão
def arvore_de_decisao():
    # 1. Carregar dados (dataset Iris)
    dados = load_iris()
    X = dados.data  # Características (features)
    y = dados.target  # Rótulos (labels)

    # 2. Dividir os dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Criar o modelo (Árvore de Decisão)
    modelo = DecisionTreeClassifier(random_state=42)

    # 4. Treinar o modelo
    modelo.fit(X_treino, y_treino)

    # 5. Fazer previsões
    previsoes = modelo.predict(X_teste)

    # 6. Avaliar o modelo
    acuracia = accuracy_score(y_teste, previsoes)
    print(f"Acurácia da Árvore de Decisão: {acuracia * 100:.2f}%")
    print("\nRelatório de Classificação:")
    print(classification_report(y_teste, previsoes, target_names=dados.target_names))

    return modelo

# Chamar a função para criar e treinar a Árvore de Decisão
modelo_arvore = arvore_de_decisao()