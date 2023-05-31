import numpy as np
# transfer function

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

test1 = sigmoidFunction(0.358)
test2 = sigmoidFunction(-1)

print(test1, test2)
