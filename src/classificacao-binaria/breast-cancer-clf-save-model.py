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

# salvando o modelo
classificador_json = classificador.to_json()
with open(PATH_DATA + '\\classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json)

## caso de algum erro para salvar este formato, tem que insatlar um novo pacote
## pip install h5py
# salvando os pesos
classificador.save(PATH_DATA + '\\classificador_breast.h5')