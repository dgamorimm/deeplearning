import numpy as np
# transfer function

def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

values = [5.0, 2.0, 1.3]

print(softmaxFunction(values))