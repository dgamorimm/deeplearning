from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os

# lendo os dados
PATH_DATA = os.path.join(os.path.dirname(__file__),'data')
base = pd.read_csv(PATH_DATA + '\\wines.csv')

# separandos os dados
X = base.iloc[:,1:14].values
y = base.iloc[:,0].values

# normalizando os dados entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0,1))
X = normalizador.fit_transform(X)