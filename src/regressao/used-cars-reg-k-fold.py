import os
import pandas as pd
from rich import print
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

base = pd.read_csv(PATH_DATA + '\\autos.csv', encoding='ISO-8859-1')

# quando você coloca axis = 1, quer dizer que você quer apagar a coluna inteira
# vamos apagar os dados que não tem relação direta com a precificação do carro
# o atributo name tem muita pluraridade nos nomes, e isso pode dificultar o modelo a encontrar um padrão
# o atributo seller e offerType esta desbalanceado , tem mais privat do que gewerblich , também foi removido
# para avaliar o desbalanceamento e os valores, podemos usar o atributo value_counts

## Pré-Processamento de dados
def drop_columns(columns : list):
    for column in columns:
        base.drop(column, axis=1, inplace=True)
    return base

def column_value_view(column : str):
    return base[column].value_counts()

def shape_view(title : str, dataframe):
    shape = dataframe.shape
    records = shape[0]
    columns = shape[1]
    print(f'Titulo: {title} || Registros: {records} || Colunas: {columns}')

def column_null_view(column : str):
    shape_view(f'Coluna Nula - {column}',base.loc[pd.isnull(base[column])])

def fill_na_more_appears(column : str):
    more_appears = column_value_view(column).head(1).index[0]
    base.fillna(value={column : more_appears}, inplace=True)

base = drop_columns(['dateCrawled', 'dateCreated', 'nrOfPictures', 
                     'postalCode', 'lastSeen', 'name', 'seller', 'offerType'])

### vimos que há preços de carros com valor 10 e até 0
### isso atrabalha a base
inconsistencia1 = base.loc[base.price <= 10]
shape_view('Inconsistencia 1', inconsistencia1)
### Essa é a correção
shape_view('Base Antes I1', base)
base = base[base.price > 10]
shape_view('Base Depois I1', base)

### vimos que há preços de carros com valor muito elevado, o que não condiz com a realidade
### isso atrabalha a base
inconsistencia2 = base.loc[base.price > 350000]
shape_view('Inconsistencia 2', inconsistencia2)
### Essa é a correção
shape_view('Base Antes I2',base)
base = base[base.price < 350000]
shape_view('Base Depois I2',base)

# essa outra inconsitencia mostra que temos dados ainda não preenchidos
# porém ela é de suma importancia pois da caracteristicas do carro
# aqui conseguimos ver a quantidade de valores nulos por coluna
colunas = [
    'vehicleType', 'gearbox', 'model', 'fuelType', 'notRepairedDamage'
]
for coluna in colunas:
    column_null_view(coluna)
# para corrigir essa inconsistencia, vamos subistituir os valores
# com que mais aparece na base
print('-------tratando a inconsistencia 3---------')
for coluna in colunas:
    fill_na_more_appears(coluna)
    column_null_view(coluna)
    

# Separando a base
previsores = base.iloc[:, 1:13].values
preco_real = base.iloc[:, 0].values

# transformando dados categoricos em valores numericos
print(previsores[0:5])  # antes
le = LabelEncoder()

colunas_categoricas = [0, 1, 3, 5, 8, 9, 10]
for idx in colunas_categoricas:
    previsores[:, idx] = le.fit_transform(previsores[:, idx])

print(previsores[0:5])  # depois


# agora temos que colocar esses dados categoricos em matrizes
# exemplo , se eu tenho uma coluna que informe o tipo de combustivel
# nessa coluna depois do label encoder, ela foi transformada em numeros 1,2,3
# o modelo preceisa enteder esses numero em matrizer da seguite forma
"""
1 - 1 0 0
2 - 0 1 0
3 - 0 0 1
"""
# vamos utiolizar o OneHotEncoder para fazer isso
# você usa esse tipo de transformação, quando o seu dado categorico não tem ordem de importancia
# exemplo, tenho carro do cambio tipo manual ou automatico, nenhum é maior ou melhor que o outro, há preferencias, ambos funcionam
onehotencoder = ColumnTransformer(
    transformers=[
        ("OneHot", OneHotEncoder(), colunas_categoricas)
        ],
    remainder='passthrough'
)
previsores = onehotencoder.fit_transform(previsores).toarray()
print(len(previsores[0]))

# função para criação da rede
def criarRede():
    # Criando a rede neural
    regressor  = Sequential()

    # para problemas de regressão é recomendado utilizar a função de ativação relu
    # units = 316 colunas da base de previsores + 1 que uma unica saida, divido por 2 = 158.5 areddondando 159
    # criando a primeira camada oculta
    regressor.add(
        Dense(
            units=158,
            activation='relu',
            input_dim = 316
        )
    )

    # criando a segunda camada oculta
    regressor.add(
        Dense(
            units=158,
            activation='relu'
        )
    )

    # criando camada de saida
    regressor.add(
        Dense(
            units=1,
            activation='linear'
        )
    )

    # quanto menor o mean_absolute_error melhor
    # eu tiver um mean_absolute_error de 2000 o meu preço pode varir 2000 para cima ou para baixo
    # compilando nosso regressor
    regressor.compile(
        loss='mean_absolute_error',
        optimizer='adam',
        metrics=['mean_absolute_error']
    )
    
    return regressor


# Criando a rede neural
regressor = KerasRegressor(
    build_fn=criarRede,
    epochs=100,
    batch_size=300
)

# realizando o treinamento com a validação cruzada
resultados = cross_val_score(
    estimator=regressor,
    X=previsores,
    y=preco_real,
    scoring='neg_mean_squared_error',
    cv=10
)

media = resultados.mean()
print("🐍 File: regressao/used-cars-reg-k-fold.py | Line: 177 | undefined ~ media",media)
desvio = resultados.std()
print("🐍 File: regressao/used-cars-reg-k-fold.py | Line: 179 | undefined ~ desvio",desvio)
