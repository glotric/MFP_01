import numpy as np
import matplotlib.pyplot as plt
from decimal import *
from scipy import special
import math
from numba import jit, prange

D = Decimal


Ai_zeros = np.zeros(100)
Bi_zeros = np.zeros(100)
Ai_zeros_ref = np.zeros(100)
Bi_zeros_ref = np.zeros(100)

Ai_zeros_ref = special.ai_zeros(100)[0]
Bi_zeros_ref = special.bi_zeros(100)[0]

def f(x):
    x = D(x)
    return ((x**D(2/3))*(D(1)+D(5/48)*(x**D(-2))-D(5/36)*(x**D(-4))+D(77125/82944)*(x**D(-6))-D(108056875/6967296)*(x**D(-8))))

for j in range(100):
    Ai_zeros[j] = -f(D(3) * D(np.pi) * (D(4)*D(j) - D(1)) / D(8))
    Bi_zeros[j] = -f(D(3) * D(np.pi) * (D(4)*D(j) - D(3)) / D(8))


Ai_zeros_err_abs = 
Bi_zeros_err_abs = 