from rbm import RBM
import numpy as np

# 6 filmes
rbm = RBM(
    num_visible= 6, # quantos nó visiveis vamos ter (quantidade de entradas)
    num_hidden=2  # camada escondida
)

# cada linha é um usuário 
# cada elemento dentro da lista é um filme
# 1 indica que ele gostou
# 0 indica que ele não gostou
# Os tres primeiros elementos são filmes de terror e os tres ultimos são de comeédia
base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

filmes = ['A Bruxa', 'Invocação do Mal', 'O Chamado',
          'Se Beber Não Case', 'Gente Grande', 'American Pie']

# treinando o meu modelo
# quanto menor o valor do erro, melhor
rbm.train(base, max_epochs=10000)

# visualizando os pesos
# a primeira linha e a primeira coluna são unidades de BIAS
print(rbm.weights)

# usuarios
usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])
usuarios = [usuario1, usuario2]

# visualizando a camada oculta para cada usuário
id_usuario = 1
for usuario in usuarios:
    camada_oculta = rbm.run_visible(usuario)  # aqui são os neuronios em que ele escolheu se vai ser filme de terror ou comédia
    recomendacao = rbm.run_hidden(camada_oculta)  # recomendação
    for i in range(len(usuario[0])):
        if usuario[0, i] == 0 and recomendacao[0, i] == 1:
            print(f'Usuário-{id_usuario}', filmes[i])
    id_usuario += 1