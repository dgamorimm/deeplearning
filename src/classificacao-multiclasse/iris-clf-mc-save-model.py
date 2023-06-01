from rich import print
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

# Lendo os dados
base = pd.read_csv(PATH_DATA + '\\iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

le = LabelEncoder()
classe = le.fit_transform(classe)

classe_dummy = np_utils.to_categorical(classe)

# Criando a rede neural
classificador = Sequential()

### Camada Oculta 1
classificador.add(
    Dense(
        units=8,
        activation='relu',
        kernel_initializer= 'random_uniform',
        input_dim = 4
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
        units=3,
        activation='softmax'
    )
)


### Compilando nossa rede neural
classificador.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

## Treinando o modelo
classificador.fit(
    previsores,
    classe_dummy,
    batch_size=10,
    epochs=1000
)

# salvando o modelo
classificador_json = classificador.to_json()
with open(PATH_DATA + '\\classificador_iris.json', 'w') as json_file:
    json_file.write(classificador_json)

## caso de algum erro para salvar este formato, tem que insatlar um novo pacote
## pip install h5py
# salvando os pesos
classificador.save(PATH_DATA + '\\classificador_iris.h5')