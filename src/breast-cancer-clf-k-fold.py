from rich import print
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
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

def criarRede():
    # Criando a rede neural
    classificador = Sequential()


    ### Camada Oculta 1
    classificador.add(
        Dense(
            units=16,
            activation='relu',
            kernel_initializer= 'random_uniform',
            input_dim = 30
        )
    )

    ### Camada Oculta 2
    classificador.add(
        Dense(
            units=16,
            activation='relu',
            kernel_initializer= 'random_uniform'
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
    otimizador = Adam(learning_rate= 0.001, decay=0.0001, clipvalue = 0.5)

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
print("🐍 File: src/breast-cancer-clf.py | Line: 9 | undefined ~ previsores",previsores.shape)
print("🐍 File: src/breast-cancer-clf.py | Line: 11 | undefined ~ classe",classe.shape)

## o parametro build_fn é justamente o que vai criar a rede neural

# Rede Neural
classificador = KerasClassifier(
    build_fn=criarRede,
    epochs=100,
    batch_size=10
)

## estimator : é o nosso classificador
## cv: é a validação cruzada, ou seja ele vai percorrer os dados de treino e teste 10 vezes
# Realizando o teste várias vezes
resultados = cross_val_score(
    estimator=classificador,
    X= previsores,
    y= classe,
    cv = 10,
    scoring='accuracy'
)

print(resultados)
# Taxa de acerto
print(resultados.mean())
# Desvio padrão, para ver o quanto cada percentual de acerto para cada treino e teste estão se desviando da média
# Quanto maior o desvio, pode se dizer que os dados se ajustam demais ao seu modelo e pode ocasionar em overfitting
print(resultados.std())