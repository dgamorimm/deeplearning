from minisom import MiniSom
from pylab import pcolor, colorbar, plot
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import pandas as pd

# utils
def graphic_view():
    plt.axis('off')
    plt.show()

# lendo os dados
PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

# Carregamento da base de dados
base1 = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
X = base1.iloc[:, 0:30].values
base2 = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')
y = base2.iloc[:,0].values

# Normalização para os dados ficarem entre 0 e 1
normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

# Aplicando a fórmula abordada no slide teórico, primeiro tiramos a raíz quadrada
# da quantidade de registros (569), que é igual a 23,85
# Multiplicamos 23,85 por 5 = 119,26
# Vamos definir o mapa auto organizável com as dimensões 11 x 11
# que equivale a 121 neurônios no total
# O input_len possui o valor 30 porque temos 30 entradas
som = MiniSom(x = 11, y = 11, input_len = 30, random_seed = 0,
              learning_rate = 0.5, sigma = 3.0)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 1000)

# O código abaixo gera o mapa auto organizável e imprime os símbolos de acordo
# com os valores das classes
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = colors[y[i]], markeredgewidth = 2)
graphic_view()