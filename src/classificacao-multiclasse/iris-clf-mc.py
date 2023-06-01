from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
import os

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

## primeiro parametro do iloc são as linhas [:,] queremos todas as linhas
## segundo parametro são as colunas [:, 0:4] queros as 4 primeiras colunas

# Lendo os dados
base = pd.read_csv(PATH_DATA + '\\iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

# como a minha classe é do nivel categórico, tremos que transformar em numeros
# essa etapa de preprocessamento tem que ser realizada para problemas assim
le = LabelEncoder()
classe = le.fit_transform(classe)

# agora a representação para cada classe fica
# iris setosa     = 1 0 0
# iris virginica  = 0 1 0
# iris versicolor = 0 0 1
classe_dummy = np_utils.to_categorical(classe)

# criação das bases de dados
previsores_treinamento,\
previsores_teste,\
classe_treinamento,\
classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

# estrutura da rede neural
classificador = Sequential()

## units = (4 colunas previsoras + 3 possiveis valores de saida) / 2 = 3.5 arrendondando 4
# 1 camada oculta
classificador.add(
    Dense(
        units=4,
        activation='relu',
        input_dim=4
    )
)

# 2 camada oculta
classificador.add(
    Dense(
        units=4,
        activation='relu'
    )
)

## tenho 3 possiveis saida (Iris-Setosa, Iris-versicolor, Iris-Virginica)
## quando trabalhamos com problema de classificação com mais de duas classes, usamos softmax
# Camada de saída
classificador.add(
    Dense(
        units=3,
        activation='softmax'
    )
)

## para modelos que tem mais de uma classe usamos é recomendado usar o categorical_crossentropy 
## tem um outro recomendado para este problema, chamado de kullback_leibler_divergence
# melhoria na descida do gradiente - otmizadores
classificador.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

# fazer o treinamento
classificador.fit(
    previsores_treinamento,
    classe_treinamento,
    batch_size=10,
    epochs=1000
)