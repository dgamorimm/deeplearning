from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


# Pré Processamento
(X_treinamento, y_treinamento),\
(X_teste, y_teste) = mnist.load_data()

previsores_treinamento = X_treinamento.reshape(
    X_treinamento.shape[0],
    28,
    28,
    1
)
previsores_treinamento = previsores_treinamento.astype('float32')
previsores_treinamento /= 255

previsores_teste = X_teste.reshape(
    X_teste.shape[0],
    28,
    28,
    1
)
previsores_teste = previsores_teste.astype('float32')
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

# Criando a rede convulucional
classificador = Sequential()
classificador.add(
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        input_shape=(28,28,1),
        activation='relu'
    )
)
classificador.add(
    MaxPooling2D(
        pool_size=(2,2)
    )
)
classificador.add(
    Flatten()
)

# Criando a rede neural densa
classificador.add(
    Dense(
        units=128,
        activation='relu'
    )
)

# Criando a camada de saída
classificador.add(
    Dense(
        units=10,
        activation='softmax'
    )
)

classificador.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

classificador.fit(
    previsores_treinamento,
    classe_treinamento,
    batch_size = 128,
    epochs = 5,
    validation_data = (previsores_teste, classe_teste)
)

resultado = classificador.evaluate(previsores_teste, classe_teste)

# Realizando a previsão

# Criamos uma única variável que armazenará a imagem a ser classificada e
# também fazemos a transformação na dimensão para o tensorflow processar
imagem_teste = X_teste[0].reshape(1, 28, 28, 1)

# Convertermos para float para em seguida podermos aplicar a normalização
imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

# Fazemos a previsão, passando como parâmetro a imagem
# Como temos um problema multiclasse e a função de ativação softmax, será
# gerada uma probabilidade para cada uma das classes. A variável previsão
# terá a dimensão 1, 10 (uma linha e dez colunas), sendo que em cada coluna
# estará o valor de probabilidade de cada classe
previsoes = classificador.predict(imagem_teste)

# Como cada índice do vetor representa um número entre 0 e 9, basta agora
# buscarmos qual é o maior índice e o retornarmos. Executando o código abaixo
# você terá o índice 7 que representa a classe 7
import numpy as np
resultado = np.argmax(previsoes)

# Caso você esteja trabalhando com a base CIFAR-10, você precisará fazer
# um comando if para indicar cada uma das classes
