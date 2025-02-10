# Importar bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Função básica para criar e treinar uma IA
def criar_e_treinar_ia():
    # 1. Carregar dados (exemplo: dataset Iris)
    dados = load_iris()
    X = dados.data  # Características (features)
    y = dados.target  # Rótulos (labels)

    # 2. Dividir os dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Criar o modelo (exemplo: Regressão Logística)
    modelo = LogisticRegression()

    # 4. Treinar o modelo
    modelo.fit(X_treino, y_treino)

    # 5. Fazer previsões
    previsoes = modelo.predict(X_teste)

    # 6. Avaliar o modelo
    acuracia = accuracy_score(y_teste, previsoes)
    print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

    return modelo

# Chamar a função para criar e treinar a IA
modelo_treinado = criar_e_treinar_ia()