from rich import print
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

def criarRede(optimizer,
              kernel_initializer,
              activation,
              neurons):
    # Criando a rede neural
    classificador = Sequential()


    ### Camada Oculta 1
    classificador.add(
        Dense(
            units=neurons,
            activation=activation,
            kernel_initializer= kernel_initializer,
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
            units=neurons,
            activation=activation,
            kernel_initializer= kernel_initializer
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
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics= ['accuracy']
        )
    
    return classificador

# Lendo os dados
base = pd.read_csv(PATH_DATA + '\\iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values


# Rede neural
classificador = KerasClassifier(
    build_fn=criarRede
)

parametros = {
    'batch_size': [30, 50, 60],
    'epochs': [1000],  
    'optimizer': ['adam', 'sgd', 'RMSprop'],  
    'kernel_initializer': ['random_uniform', 'random_normal','normal'], 
    'activation': ['selu', 'relu', 'elu'], 
    'neurons' : [4, 2]
}

grid_search = GridSearchCV(
    estimator=classificador,
    param_grid=parametros,
    scoring='accuracy',
    cv=10
)

grid_search = grid_search.fit(previsores, classe)

melhores_parametros =  grid_search.best_params_
# {
#     'activation': 'relu',
#     'batch_size': 30,
#     'epochs': 100,
#     'kernel_initializer': 'random_uniform',
#     'loss': 'binary_crossentropy',
#     'neurons': 8,
#     'optimizer': 'adam'
# }
print("🐍 File: src/breast-cancer-clf-tuning.py | Line: 116 | undefined ~ melhores_parametros",melhores_parametros)

melhor_precisao = grid_search.best_score_
# 0.9104331625523987
print("🐍 File: src/breast-cancer-clf-tuning.py | Line: 119 | undefined ~ melhor_precisao",melhor_precisao)