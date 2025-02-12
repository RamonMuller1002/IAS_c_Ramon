import panda as pd
arquivo = pd.read.csv('C:/DT/wine_dataset.csv')

arquivo['style'] = arquivo['style'].replace('red', 0)
arquivo['style'] = arquivo['style'].replace('white', 1)

y = arquivo['style']
x = arquivo.drop('style', axis = 1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste