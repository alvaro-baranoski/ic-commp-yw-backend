# Testing AR coefficients estimation with YW method
import random
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import yule_walker

# Creating input data
# y(k) = 1.5*y(k-1) - 0.7y(k-2) + e(k)
dt = 1e-3

Ndata = 1000
k = [i for i in range(Ndata)]
e = [random.random() for _ in range(Ndata)]
y = [i for i in range(Ndata)]

y[0] = e[0]
y[1] = 1.5 * y[0] + e[1]

for index in range(2, Ndata):
    y[index] = 1.5 * y[index - 1] - 0.7 * y[index - 2] + e[index]

# Data processing
y -= np.mean(y)

# modelo
# y(k) = [ y(k-1) y(k-2) ]*[ a1 ] + e(k)
#                          [ a2 ]

order = 2
ar, sigma = yule_walker(y, order=order)
ar *= -1

coeff = np.array([1])
coeff = np.append(coeff, ar)

print("modos estimados")
raizes_est_z = np.roots(coeff)
raizes_est_s = np.log(raizes_est_z) / dt
print(raizes_est_s)

print("modos reais")
raizes_reais_z = np.roots([1, -1.5, 0.7])
raizes_reais_s = np.log(raizes_reais_z) / dt
print(raizes_reais_s)
