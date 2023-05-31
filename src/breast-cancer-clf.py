from rich import print
from sklearn.model_selection import train_test_split
import pandas as pd
import os

PATH_DATA = os.path.join(os.path.dirname(__file__),'data')

previsores = pd.read_csv(PATH_DATA + '\\entradas_breast.csv')
classe = pd.read_csv(PATH_DATA + '\\saidas_breast.csv')

print('Previsores[shape]: ',previsores.shape)
print('Classes[shape]: ',classe.shape)

previsores_treinamento,\
previsores_teste,\
classe_treinamento,\
classe_teste = train_test_split(previsores, classe, test_size=0.25)

print('\n')
print('Previsores Treinamento[shape]: ',previsores_treinamento.shape)
print('Previsores Teste[shape]: ',previsores_teste.shape)
print('Classe Treinamento[shape]: ',classe_treinamento.shape)
print('Classe Teste[shape]: ',classe_teste.shape)

