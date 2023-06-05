from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
from rich import print
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# lendo os dados
PATH_DATA = os.path.join(os.path.dirname(__file__),'data')
base = pd.read_csv(PATH_DATA + '\\petr4_treinamento.csv')

#utils
def shape_view(title : str, dataframe):
    shape = dataframe.shape
    c = 17
    c2 = 7
    c3 = 5
    try:
        records = shape[0]
        pixel_h = shape[1]
        pixel_v = shape[2]
        channel = shape[3]
        total = pixel_h * pixel_v
        print(f'Titulo: {title}{(c-len(title))*" "}|| Registros: {records}{(c2-len(str(records)))*" "}|| Pixels: {pixel_h} X {pixel_h} = {total}{(c3-len(str(total)))*" "}|| Canais: {channel}')
    except:
        try:
            records = shape[0]
            pixel_h = shape[1]
            pixel_v = shape[2]
            total = pixel_h * pixel_v
            print(f'Titulo: {title}{(c-len(title))*" "}|| Registros: {records}{(c2-len(str(records)))*" "}|| Pixels: {pixel_h} X {pixel_h} = {total}')
        except:
            records = shape[0]
            try:
                columns = shape[1]
            except:
                columns = 0
            print(f'Titulo: {title}{(c-len(title))*" "}|| Registros: {records}{(c2-len(str(records)))*" "}|| Colunas: {columns}')
            
def column_null_view(column : str):
    shape_view(f'Coluna Nula - {column}',base.loc[pd.isnull(base[column])])
    
def drop_columns(columns : list):
    for column in columns:
        base.drop(column, axis=1, inplace=True)
    return base

def column_value_view(column : str):
    return base[column].value_counts()

def drop_lines_na():
    # axis = 0 ele apaga a linha 
    base.dropna(axis=0, inplace=True)

def graphic_view():
    plt.axis('off')
    plt.show()

# pré processamento
shape_view('Inicio Base', base)
drop_lines_na()
shape_view('S/ Registros Nulos', base)

## selecionando os dados de treinamento
base_treinamento = base.iloc[:, 1:2].values
print('Valores Reais\n',base_treinamento[:5])
base_valor_maximo = base.iloc[:, 2:3].values
print('Valores Maximo\n',base_valor_maximo[:5])


## como tulizando varias camadas, para melhorar o processamento, é bom que não utlizemos os valores reaisn  e sim normalizados
## vamos colocar os valores em uma escala de 0 até 1
normalizador = MinMaxScaler(
    feature_range=(0, 1)
)
base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)
print('Treinamento Normalizado\n',base_treinamento_normalizada[:5])
base_valor_maximo_normalizada = normalizador.fit_transform(base_valor_maximo)
print('Valor Máx Normalizado\n',base_valor_maximo_normalizada[:5])


## armazenado 90 valores anteriores e o preço real
previsores = []
preco_real1 = []
preco_real2 = []

for i in range(90, base.shape[0]):
    previsores.append(base_treinamento_normalizada[i-90:i, 0])
    preco_real1.append(base_treinamento_normalizada[i, 0])
    preco_real2.append(base_valor_maximo_normalizada[i, 0])

## adicionando a um array do numpy, pois o keras só entende assim
previsores, preco_real1, preco_real2 = np.array(previsores), np.array(preco_real1), np.array(preco_real2)

previsores = np.reshape(
    previsores,
    (
        previsores.shape[0],  # quantiodade de registros
        previsores.shape[1],  # quantidade de tempo
        1  #trabalhamos com apenas um idicador, nesse caso estamos trabalhando somente com a coluna "Open"
    )  
)

# junta as duas colunas uma do lado da outra
preco_real =  np.column_stack((preco_real1, preco_real2))

print('Previsores (90 dias anteriores):\n',previsores[:5])
print('Preco real:\n',preco_real[:5])


# criando a estrutura da rede neural
regressor = Sequential()

## units = tem que ser um valor alto. aqui define a quantidade de células de memória
## return_sequences = True define que a sua camada vai passar informação para outra camada LSTM, vai jogar a informação para frente e para outras camadas subsequentes
## o numero 6 indica temos mais de um atributo previsor
## para essa rede neural é interessante colocar mais camadas, pois poucas camadas o resultado não é interessante

# criando a primeira camada com células de memória
regressor.add(
    LSTM(
        units=100,
        return_sequences= True,
        input_shape = (previsores.shape[1], 1)
    )
)
regressor.add(
    Dropout(
        0.3
    )
)

# criando a segunda camada com células de memória
regressor.add(
    LSTM(
        units=50,
        return_sequences= True
    )
)
regressor.add(
    Dropout(
        0.3
    )
)

# criando a terceira camada com células de memória
regressor.add(
    LSTM(
        units=50,
        return_sequences= True
    )
)
regressor.add(
    Dropout(
        0.3
    )
)

# criando a quarta camada com células de memória
regressor.add(
    LSTM(
        units=50
    )
)
regressor.add(
    Dropout(
        0.3
    )
)

# criando a camada de saída (testar com a funçaõ linear)
regressor.add(
    Dense(
        units=2,
        activation='linear'
    )
)

## é recomendado o rmsprop, mas o adam também da bons resultados
# otimizando a descida do gradiente
regressor.compile(
    optimizer='rmsprop',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

## recomendado rodar 100 ou mais épocas para este tipo de algoritmo
# treinando o modelo
regressor.fit(
    previsores,
    preco_real,
    epochs=100,
    batch_size=32
)

# estruturando a base de teste
base_teste = pd.read_csv(PATH_DATA + '\\petr4_teste.csv')
preco_real_open = base_teste.iloc[:, 1:2].values
preco_real_high = base_teste.iloc[:, 2:3].values

## pegar os 90 preços anteriores
### tem que concatenar com a base inicial para pegar os valores anterios
base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0)
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for  i in range(90, len(entradas)):
    X_teste.append(entradas[i-90:i,0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0],X_teste.shape[1],1))


# realizando a previsão
previsoes = regressor.predict(X_teste)

# olhar de uma forma melhor desnormalizando o dado
previsoes = normalizador.inverse_transform(previsoes)

# Avaliando a média entre elas
print('Previsao: ', previsoes.mean())
print('Preço Real: ', preco_real_open.mean())

# olhando no gráfico
plt.plot(preco_real_open, color = 'red', label='Preço Abertura Real')
plt.plot(preco_real_high, color = 'black', label='Preço Alta Real')

plt.plot(previsoes[:, 0], color = 'blue', label='Previsões Abertura')
plt.plot(previsoes[:, 1], color = 'orange', label='Previsões Alta')

plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()

graphic_view()