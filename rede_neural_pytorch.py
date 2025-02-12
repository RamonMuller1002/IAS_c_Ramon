import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Carregar e pré-processar o dataset MNIST
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
treino_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
teste_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

treino_loader = DataLoader(treino_dataset, batch_size=64, shuffle=True)
teste_loader = DataLoader(teste_dataset, batch_size=64, shuffle=False)

# 2. Definir o modelo de rede neural
class RedeSimples(nn.Module):
    def __init__(self):
        super(RedeSimples, self).__init__()
        self.flatten = nn.Flatten()
        self.camada1 = nn.Linear(28 * 28, 128)
        self.camada2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.camada1(x))
        x = self.camada2(x)
        return x

modelo = RedeSimples()

# 3. Definir função de perda e otimizador
criterio = nn.CrossEntropyLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

# 4. Treinar o modelo
for epoca in range(5):
    for imagens, rotulos in treino_loader:
        saidas = modelo(imagens)
        perda = criterio(saidas, rotulos)
        otimizador.zero_grad()
        perda.backward()
        otimizador.step()
    print(f"Época {epoca + 1}, Perda: {perda.item()}")

# 5. Avaliar o modelo
modelo.eval()
corretos = 0
total = 0
with torch.no_grad():
    for imagens, rotulos in teste_loader:
        saidas = modelo(imagens)
        _, predicoes = torch.max(saidas, 1)
        total += rotulos.size(0)
        corretos += (predicoes == rotulos).sum().item()

print(f"Acurácia no teste: {100 * corretos / total:.2f}%")