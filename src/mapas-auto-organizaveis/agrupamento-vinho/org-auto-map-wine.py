from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot
import matplotlib.pyplot as plt
import pandas as pd
import os

# utils
def graphic_view():
    plt.axis('off')
    plt.show()

# lendo os dados
PATH_DATA = os.path.join(os.path.dirname(__file__),'data')
base = pd.read_csv(PATH_DATA + '\\wines.csv')

# separandos os dados
X = base.iloc[:,1:14].values
y = base.iloc[:,0].values

# normalizando os dados entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)

# Achando x e y:
# minha base tem 178 registros
# 5 x raiz quadrada de 178 = 65,5 aproximado a 64 , ou seja , 8x8 = 64
# construindo nosso mapa organizavel
som = MiniSom(
    x = 8, # quantas linhas meu mapa vai ter
    y = 8, # quantas colunas meu mapa vai ter
    input_len=13,  #tamanho da coluna de X
    sigma=1.0,  # equivale ao raio ou alcance dos meus neuronios no meu BMU
    learning_rate=0.5,  # taxa de aprendizagem e atualização dos pesos
    random_seed= 2  # para obter o mesmo resultado da aula
)

# inicialização dos pesos
som.random_weights_init(X)

#treinando o mapa
som.train_random(
    data=X, # base de dados
    num_iteration=100  # equivalente as épocas
)

# visualizando os pesos dos centroides
print(som._weights)

# visualizando os pesos dos mapas organizaveis
print(som._activation_map)

# visualizando os neuronios que mais tiveram a seleção como BMU
q = som.activation_response(X)
print(q)

# visualizando o calculo do MID = mean inter neuron distance (Média da Distancia Euclidiana)
# conseguimos ver a media de distancia dos seus vizinhos
pcolor(som.distance_map().T)
colorbar()
# quanto mais escuro for, mais parecido com seus vizinhos ele é
# quanto mais claro for, mais diferente este neuronio é dos seus vizinhos

# esse aqui vai dizer, qual o neuronio ganhador de cada um dos registros
w = som.winner(X[1])  #BMU do dado na posição 1

# criando os marcadores
markers = ['o', 's', 'D']

# criando as cores
color = ['r', 'g', 'b']

# mudando os indices dos grupos
y[y == 1] = 0
y[y == 2] = 1
y[y == 3] = 2

# percorrendo todos os registros
for i, x in enumerate(X):
    w = som.winner(x)
    plot(
        w[0] + 0.5, #o + 0.5 serve para posicionar o simbulo centralizado 
        w[1] + 0.5, #o + 0.5 serve para posicionar o simbulo centralizado
        markers[y[i]],  # vai colocar o marcador de acordo com a classe
        markerfacecolor = 'None',  # cor da fonte
        markersize=10,  # tamanho do marcador
        markeredgecolor=color[y[i]],  # preencher a cor do marcador de com a classe
        markeredgewidth=2  # configurarmos o tamanho da borda
    )
graphic_view()