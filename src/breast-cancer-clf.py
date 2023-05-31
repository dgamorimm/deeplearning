from rich import print
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os

##### Anotações ######
# O modelo é Sequential justamente poque ele tem a camada de entrada, camda oculta e de saída
# Vamos utilizar camadas Dense(Densas) em uma rede neural. Siginifica que cada um dos neuronios é ligado na camada subsequente

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

# Lendo os dados dos previsores e das classes
previsores = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
classe = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')
print("🐍 File: src/breast-cancer-clf.py | Line: 9 | undefined ~ previsores",previsores.shape)
print("🐍 File: src/breast-cancer-clf.py | Line: 11 | undefined ~ classe",classe.shape)

# Separando dados de teste e treino
previsores_treinamento,\
previsores_teste,\
classe_treinamento,\
classe_teste = train_test_split(previsores, classe, test_size=0.25)

print('\n')
print("🐍 File: src/breast-cancer-clf.py | Line: 15 | undefined ~ previsores_treinamento",previsores_treinamento.shape)
print("🐍 File: src/breast-cancer-clf.py | Line: 16 | undefined ~ previsores_teste",previsores_teste.shape)
print("🐍 File: src/breast-cancer-clf.py | Line: 17 | undefined ~ classe_treinamento",classe_treinamento.shape)
print("🐍 File: src/breast-cancer-clf.py | Line: 18 | undefined ~ classe_teste",classe_teste.shape)

# Criando a rede neural
classificador = Sequential()

## Criando a primeira camada oculta
## no keras não definimos a quantidade da camada de entrada, ja iniciamos a partir da camada oculta
## para sabermos a quantidade de neuronios na camada oculta podemos fazer a seguinte formula
## units = quantidadeEntrada + quantidadeSaida / 2
## quantidadeEntrada = 30, pois no nossa base de dados previsores, possui 30 colunas
## quantidadeSaida = Como queremos saber se maligno ou bnigno, existe apenas uma unica saida, então numero 1
## calculo ---> units = (30 + 1) / 2 =  15.5 arredondar para cima 16
## usando a função de ativação relu por recomendação de boas práticas comerciais, ela tende a dar mais certo do que a sigmoid, hiperbolice e etc.
## kernel_initializar = é como você vai inicializar o balancemaento dos pesos, por padrão , utilizamos o random uniform
## input_dim  = define quantas entradas você tem no seu modelo. Como vimos anteriormente temos 30. Define 30 neuronios na entrada

### Camada Oculta
classificador.add(
    Dense(
        units=16,
        activation='relu',
        kernel_initializer= 'random_uniform',
        input_dim = 30
    )
)


## como só temos uma possivel saida , Benigno ou Maligno, então utilizaremos units = 1
## utilizaremos a função de ativação sigmoid, pois ela retorna valores entre 0 e 1, que é o nosso caso, poistemos uma resposta binária

### Camada de Saida
classificador.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)


## optimizer: qual pe a função para fazer o ajuste dos pesos, entra na parte da descida do gradiente + descida do gradiente estocastico
## é recomendado que começamos a utlizar o optimizer 'adam', é o que melhor se adpta na maioria dos casos
## loss: é a nossa função de perda, a mneira que vai fazer o tratamento ou o calculo do erro
## quanto menor o loss, melhor
## como temos um problema de classificação binário, é recomendado que utilize o 'binary_crossentropy'
## é recomendado usar o crossentropy , pois ele usa o logaritmo, ele vai de certa forma vai utilizar a perda através de uma regressão logistica
## usamos a métrica de binary_accuracy, pois estamos com problema de classificação binária

### Compilando nossa rede neural
classificador.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics= ['binary_accuracy']
    )

## fit: significa achar a correlação entre os valores de treinamento
## batch_size : vai calcular o erro para 10 registros e depois ele vai atualizar os pesos
## epochs : quantas vezes eu quero fazer os ajustes dos pesos
### Treinando
classificador.fit(
    previsores_treinamento,
    classe_treinamento,
    batch_size=10,
    epochs=100
)