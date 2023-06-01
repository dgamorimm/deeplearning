import numpy as np
# transfer function

def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0

test1 = reluFunction(0.358)
test2 = reluFunction(-1)

print(test1, test2)