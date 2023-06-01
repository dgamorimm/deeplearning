from keras.models import model_from_json
from rich import print
import numpy as np
import os
import pandas as pd

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

## usamos o método close do arquivo para liberar espaço em memória

# Carregando a estrutura de classificação 
arquivo = open(PATH_DATA + '\\classificador_breast.json', 'r')
estrutura_rede = arquivo.read()
arquivo.close()

# carregando o modelo
classificador = model_from_json(estrutura_rede)

# carregando os pesos
classificador.load_weights(PATH_DATA + '\\classificador_breast.h5')

# fazendo uma previsão para testar

## criando um registro unico
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08,
                  0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500,
                  145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007,
                  23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84,
                  158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)
print("🐍 File: src/breast-cancer-clf-load-model.py | Line: 31 | undefined ~ previsao",previsao)

# mostrando o resultado da acurácia e perda compilando o classificador

## Lendo os dados dos previsores e das classes
previsores = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
classe = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')

## compilando
classificador.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['binary_accuracy']
)

## verificando a acurácia e a perda
resultado = classificador.evaluate(previsores, classe)
print("🐍 File: src/breast-cancer-clf-load-model.py | Line: 49 | undefined ~ resultado",resultado)
