# Importar bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

# Função para criar e treinar uma Rede Neural
def rede_neural():
    # 1. Carregar dados (dataset Iris)
    dados = load_iris()
    X = dados.data  # Características (features)
    y = dados.target  # Rótulos (labels)

    # 2. Codificar os rótulos em one-hot encoding
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))  # Redimensionar y para 2D

    # 3. Dividir os dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Criar o modelo (Rede Neural)
    modelo = Sequential([
        Dense(10, input_shape=(4,), activation='relu'),  # Camada oculta com 10 neurônios
        Dense(10, activation='relu'),  # Outra camada oculta
        Dense(3, activation='softmax')  # Camada de saída com 3 neurônios (uma para cada classe)
    ])

    # 5. Compilar o modelo
    modelo.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

    # 6. Treinar o modelo
    modelo.fit(X_treino, y_treino, epochs=50, batch_size=8, verbose=1)

    # 7. Fazer previsões
    previsoes = modelo.predict(X_teste)
    previsoes_classes = previsoes.argmax(axis=1)  # Converter one-hot encoding de volta para classes
    y_teste_classes = y_teste.argmax(axis=1)

    # 8. Avaliar o modelo
    acuracia = accuracy_score(y_teste_classes, previsoes_classes)
    print(f"Acurácia da Rede Neural: {acuracia * 100:.2f}%")

    return modelo

# Chamar a função para criar e treinar a Rede Neural
modelo_rede_neural = rede_neural()