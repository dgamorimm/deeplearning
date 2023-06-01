from rich import print
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
import os
import pandas as pd

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

def criarRede():
    # Criando a rede neural
    classificador = Sequential()


    ### Camada Oculta 1
    classificador.add(
        Dense(
            units=32,
            activation='selu',
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
            units=16,
            activation='selu',
            kernel_initializer= 'random_uniform'
        )
    )
    
     ### Tecnica de Dropout para mitigar Overfitting Camada Oculta 2
    classificador.add(
        Dropout(
           0.13
        )
    )
    
    
    ### Camada de Saida
    classificador.add(
        Dense(
            units=1,
            activation='sigmoid'
        )
    )

    ### Criando um otimizador parametrizavel
    otimizador = Adam(learning_rate= 0.003, decay=0.0003, clipvalue = 0.8)

    ### Compilando nossa rede neural
    classificador.compile(
        optimizer=otimizador,
        loss='binary_crossentropy',
        metrics= ['binary_accuracy']
        )
    
    return classificador

# Lendo os dados dos previsores e das classes
previsores = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
classe = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')

# Rede Neural
classificador = KerasClassifier(
    build_fn=criarRede,
    epochs=120,
    batch_size=20
)

# Realizando o teste várias vezes
resultados = cross_val_score(
    estimator=classificador,
    X= previsores,
    y= classe,
    cv = 5,
    scoring='accuracy'
)

print(resultados.mean())
print(resultados.std())