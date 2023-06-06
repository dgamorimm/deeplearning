import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, Input
from keras.models import Model, Sequential

#utils
def graphic_view():
    plt.axis('off')
    plt.show()

# carregando as bases de dados
# como ele vai fazer o encode e o decode, não vamos utilizar base de teste, tendo em vista que a mesa entrada será a mesma saída
(previsores_treinamento, _),\
(previsores_teste, _) = mnist.load_data()
# dim treinamento = (60000, 28, 28) = 28 x 28 = 784 Pixels
# dim teste = (10000, 28, 28) = 28 x 28 = 784 Pixels

# normalizando o dado entre 0 e 1, mas poderiamos usar o MinMaxScaler
previsores_treinamento = previsores_treinamento.astype('float32') / 255
previsores_teste = previsores_teste.astype('float32') / 255

# adicionado dimensões na base
# 60000 registros, por 784 pixels
# cada uma dessas linhas nós temos uma imagem
previsores_treinamento = previsores_treinamento.reshape(
    (len(previsores_treinamento), np.prod(previsores_treinamento.shape[1:]))
)
# adicionado dimensões na base
# 10000 registros, por 784 pixels
# cada uma dessas linhas nós temos uma imagem
previsores_teste = previsores_teste.reshape(
    (len(previsores_teste), np.prod(previsores_teste.shape[1:]))
)

# Entrada : 784
# Neuronio da camada oculta: 32
# Fator de compactação: 784 / 32 = 24.5
# Saida : 784
# criando o autoencoder

autoencoder = Sequential()

# por padrão a relu da bons resultados
# camada de entrada com a camada oculta / encode
autoencoder.add(
    Dense(
        units=32,
        activation='relu',
        input_dim= 784
    )
)

# podemos usar sigmoid nesse caso, pois deixamos os dados normalizados
# muito comum usar a tangente hiperbolica
# camada de saida / reconstrução / decode
autoencoder.add(
    Dense(
        units=784,
        activation='sigmoid'
    )
)

# mostrando a estrutura da rede neural
print(autoencoder.summary())

# compilando o modelo
autoencoder.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# como a saida tem que ser igual a entrada o terinamento será com o mesmo previsor
# treinando o modelo
autoencoder.fit(
    previsores_treinamento,
    previsores_treinamento,
    epochs=50,
    batch_size=256,
    validation_data = (previsores_teste, previsores_teste)
)

# codificando e decodificando para gerar a imagem
dimensao_original = Input(
    shape=(784,)
)

# primeira camada que nós contruimos acima
camada_encoder = autoencoder.layers[0]
encoder = Model(
    dimensao_original,
    camada_encoder(dimensao_original)
)
encoder.summary()

# criando as imagens codificadas | reduzindo dimensionalidade para 32 da camada oculta
imagens_codificadas = encoder.predict(previsores_teste)

# criando as imagens decodificadas | retornando os 784 para a camada de saida
imagens_decodificadas = autoencoder.predict(previsores_teste)

# visualizando os resultados
# capturar 10 numeros aleatórios de 0 a 10000 registros
numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size=numero_imagens)

# criando o gráfico
plt.figure(figsize=(18, 18))
for i, idx_img in enumerate(imagens_teste):
    # imagem original
    eixo = plt.subplot(10, 10, i +1)
    plt.imshow(previsores_teste[idx_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
    
    # imagem codificada
    eixo = plt.subplot(10, 10, i +1 + numero_imagens)
    plt.imshow(imagens_codificadas[idx_img].reshape(8,4)) # 8 * 4 = 32
    plt.xticks(())
    plt.yticks(())
    
    # imagem decodifica/reconstruida
    eixo = plt.subplot(10, 10, i +1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[idx_img].reshape(28,28))
    plt.xticks(())
    plt.yticks(())
graphic_view()