from rich import print
from keras.models import Sequential
from keras.layers import Dense, Dropout
import os
import pandas as pd
import numpy as np

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

# Lendo os dados dos previsores e das classes
previsores = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
classe = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')

# Criando a rede neural
classificador = Sequential()

### Camada Oculta 1
classificador.add(
    Dense(
        units=8,
        activation='relu',
        kernel_initializer= 'random_uniform',
        input_dim = 30
    )
)

### Tecnica de Dropout para mitigar Overfitting Camada Oculta 1
classificador.add(
    Dropout(
        0.2
    )
)

### Camada Oculta 2
classificador.add(
    Dense(
        units=8,
        activation='relu',
        kernel_initializer= 'random_uniform'
    )
)

### Tecnica de Dropout para mitigar Overfitting Camada Oculta 2
classificador.add(
    Dropout(
        0.2
    )
)

### Camada de Saida
classificador.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)


### Compilando nossa rede neural
classificador.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics= ['binary_accuracy']
    )

## Treinando o modelo
classificador.fit(
    previsores,
    classe,
    batch_size=30,
    epochs=100
)

# temos que colocar uma lista sobre outra pois ela é um registro e se identifica com uma unica linha
# os valores colocados é com base nos dados de input, ou seja, queremos prever através dessas novas informações aleatórias se o cancer é maligno ou benigno

## criando um registro unico
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08,
                  0.134, 0.178, 0.20, 0.05, 1098, 0.87, 4500,
                  145.2, 0.005, 0.04, 0.05, 0.015, 0.03, 0.007,
                  23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84,
                  158, 0.363]])
print("🐍 File: src/breast-cancer-clf-one-record.py | Line: 94 | undefined ~ novo",novo.shape)

## realizando a previsão
previsao = classificador.predict(novo)
print("🐍 File: src/breast-cancer-clf-one-record.py | Line: 102 | undefined ~ previsao",previsao)
previsao = (previsao > 0.5)
print("\n 🐍 File: src/breast-cancer-clf-one-record.py | Line: 104 | undefined ~ previsao",previsao)