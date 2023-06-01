from rich import print
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

def save_txt(predictions):
    # Criar data frame
    df = pd.DataFrame(predictions)
    
    # Formatando os valores sem notação científica
    df_formatted = df.apply(lambda x: '{:.0f}'.format(x[0]), axis=1)
    
    # Salvando o DataFrame formatado em um arquivo de texto
    df_formatted.to_csv(PATH_DATA + '\\classe_prevista.txt', index=False, header=False)

def criarRede(optimizer,
              loss,
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
            input_dim = 30
        )
    )
    
    ## essa tecnica zera o valor de alguns neuronios através de um percentual definido, afim de mitigar o Overfitting
    ## é usado depois das camdas que são implementadas
    ## é bom adicionar para melhorar os resultados
    
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
            units=1,
            activation='sigmoid'
        )
    )
    

    ### Compilando nossa rede neural
    classificador.compile(
        optimizer=optimizer,
        loss=loss,
        metrics= ['binary_accuracy']
        )
    
    return classificador

# Lendo os dados dos previsores e das classes
previsores = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
classe = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')
print("🐍 File: src/breast-cancer-clf.py | Line: 9 | undefined ~ previsores",previsores.shape)
print("🐍 File: src/breast-cancer-clf.py | Line: 11 | undefined ~ classe",classe.shape)

# Rede neural
classificador = KerasClassifier(
    build_fn=criarRede
)

parametros = {
    'batch_size': [10, 30],  # descida do gradiente estocastica
    'epochs': [50, 100],  # quantidade de vezes (interessante valores maiores)
    'optimizer': ['adam', 'sgd'],  # o sgd é inferior ao adam
    'loss': ['binary_crossentropy', 'hinge'],  # hinge é inferrior ao entropy em termos de perda
    'kernel_initializer': ['random_uniform', 'normal'],  # distribuição normal estatisca na distribuição dos pesos
    'activation': ['relu', 'tanh'],  # relu da melhores resultados, e a hiperbólica
    'neurons' : [16, 8]  # quantidade de neuronios na camada oculta (interessante valores maiores) 
}

grid_search = GridSearchCV(
    estimator=classificador,
    param_grid=parametros,
    scoring='accuracy',
    cv=5
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