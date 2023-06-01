import numpy as np
# transfer function

def tangetFunction(soma):
    nominator = (np.exp(soma) - np.exp(-soma))
    denominator = (np.exp(soma) + np.exp(-soma))
    return nominator / denominator

test1 = tangetFunction(0.358)
test2 = tangetFunction(-0.358)

print(test1, test2)