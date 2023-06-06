from rbm import RBM
import numpy as np

# A RBM é criada com seis nós visíveis, que equivalem a camada de entrada e a 
# quantidade de filmes na base de dados. Como neste exemplo os filmes possuem
# o estilo mais variado do que o exemplo de terror e comédia, foram definidos
# três neurônios na camada oculta para aumentar a diversidade de características
# a serem capturadas
rbm = RBM(num_visible = 6, num_hidden = 3)

# Criação da base de dados conforme o slide com as notas, com exceção do usuário
# Leonardo que será o alvo das recomendações. Lembrando que aqui colocamos
# 1 se o usuário assitiu e 0 caso tenha outra resposta
base = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1], 
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])

# Cadastro dos filmes conforme o slide
filmes = ["Freddy x Jason", "O Ultimato Bourne", "Star Trek", 
          "Exterminador do Futuro", "Norbit", "Star Wars"]

# Treinamento da RBM
rbm.train(base, max_epochs = 5000) 

# Criação do registro que corresponde aos filmes do Leonardo, conforme o slide
leonardo = np.array([[0,1,0,1,0,0]]) 

# Variável que recebe quais dos três neurônios da camada oculta foram ativados
camada_escondida = rbm.run_visible(leonardo)

# Faz a recomendação e imprime o nome dos filmes
recomendacao = rbm.run_hidden(camada_escondida)
for i in range(len(leonardo[0])):
    if leonardo[0, i] == 0 and recomendacao[0, i] == 1:
        print(filmes[i])
        
# No exemplo utilizando filtragem colaborativa o filme recomendado também
# foi o Star Wars, conforme a recomendação utilizando RBM