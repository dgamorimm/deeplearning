import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

# utils
def image_view():
    plt.axis('off')
    plt.show()

# carregando a nossa base de dados
base =  datasets.load_digits()

# separando a base de dados
previsores = np.asarray(base.data, 'float32')
classe = base.target

# normalizando o dado entre 0 e 1
normalizador = MinMaxScaler(feature_range=(0,1))
previsores = normalizador.fit_transform(previsores)

# dividindo a base para fazer o trinamento e depois testar
previsores_treinamento,\
previsores_teste,\
classe_treinamento,\
classe_teste =train_test_split(
    previsores,
    classe,
    test_size=0.2,
    random_state=0
)

# aplicando a redução de dimensionalidade

# iniciando o algoritmo rbm
rbm = BernoulliRBM(random_state=0)
# adicionando o numero de épocas
rbm.n_iter = 25
# numero de componentes que vamos adicionar( Numeros de neuronios na camada escondida)
# é recomendado fazer o tuning dos parametros para saber a melhor quantidade a ser colocada
# a minha base tem 64 caracteristicas e estou reduzindo a 50
rbm.n_components = 50

# iniciado o algoritmo de probabilidades
naive_rbm = GaussianNB()

# criando as etpas de execução da rede neural
classificador_rbm = Pipeline(
    steps=(
        [
            ('rbm', rbm),
            ('naive', naive_rbm)
        ]
    ),
    verbose=True
)

# treinando o modelo
classificador_rbm.fit(previsores_treinamento, classe_treinamento)

# visualizando as imagens
plt.figure(figsize=(20,20))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8,8)), cmap=plt.cm.gray_r)
    plt.xticks(())
    plt.yticks(())
image_view()