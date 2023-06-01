from keras.models import model_from_json
from rich import print
import numpy as np
import os

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

## usamos o método close do arquivo para liberar espaço em memória

# Carregando a estrutura de classificação 
arquivo = open(PATH_DATA + '\\classificador_iris.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

# carregando o modelo
classificador = model_from_json(estrutura_rede)

# carregando os pesos
classificador.load_weights(PATH_DATA + '\\classificador_iris.h5')

# fazendo uma previsão para testar

# Criar e classificar novo registro
novo = np.array([[3.2, 4.5, 0.9, 1.1]])

#executando a previsao
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

# mostrando a saida
if previsao[0][0] == True and previsao[0][1] == False and previsao[0][2] == False:
    print('Iris setosa')
elif previsao[0][0] == False and previsao[0][1] == True and previsao[0][2] == False:
    print('Iris virginica')
elif previsao[0][0] == False and previsao[0][1] == False and previsao[0][2] == True:
    print('Iris versicolor')