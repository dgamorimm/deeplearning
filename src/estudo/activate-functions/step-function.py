import numpy as np
# transfer function

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

test1 = stepFunction(30)
test2 = stepFunction(-1)

print(test1, test2)
