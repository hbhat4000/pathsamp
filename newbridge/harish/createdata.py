import numpy as np
import scipy.special as poly
from math import factorial

# these parameters define the model
# coef essentially defines the drift function
coef = np.array([1., 2., -1., -0.5])
# this is the diffusion coefficient
g = 1/2

def mypoly(x, dof):
    y = np.zeros((x.size, dof))
    for i in range(x.size):
        for j in range(dof):
            H = poly.hermitenorm(j, monic = True)
            # normalization constant = np.sqrt(np.sqrt(2 * np.pi) * factorial(j))
            # for probabilist's Hermite (hermitenorm)
            # normalization constant = np.sqrt(np.sqrt(np.pi) * (2**j) * factorial(j))
            # for physicists' Hermite (hermite)
            y[i,j] = H(x[i]) / np.sqrt(np.sqrt(2 * np.pi) * factorial(j))
    return y

def drift(x, theta):
    dof = theta.shape[0]
    return (np.dot(mypoly(x, dof), theta))

numsteps = 25000
savesteps = 10
ft = 10.0
ic = np.array([1.0])
it = np.array([0.0])

def createpath():
    j = 1
    x = np.zeros(savesteps + 1)
    t = np.zeros(savesteps + 1)
    h = ft / numsteps
    h12 = np.sqrt(h)
    x[0] = ic
    t[0] = it
    for i in range(1, numsteps + 1):
        ic = ic + drift(ic, coef) * h + g * h12 * np.random.standard_normal(1)
        if (i % (numsteps / savesteps) == 0):
            x[j] = ic
            t[j] = i*h + it
            j = j + 1


