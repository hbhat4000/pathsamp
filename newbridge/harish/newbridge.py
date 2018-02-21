import numpy as np
import scipy.special as poly
from math import factorial

# this function defines our basis functions
# x must be a numpy array, a column vector of points
# (x = vector of points at which we seek to evaluate the basis functions)
# dof is the number of degrees of freedom, i.e., the number of basis functions
def mypoly(x, dof):
    y = np.zeros((x.shape[0], dof))
    for i in range(x.shape[0]):
        for j in range(dof):
            H = poly.hermitenorm(j, monic = True)
            y[i,j] = H(x[i]) / np.sqrt(np.sqrt(2 * np.pi) * factorial(j))
    return y

# drift function using "basis" functions defined by mypoly
# x must be a numpy array, the points at which the drift is to be evaluated
# theta must also be a numpy array, the coefficients of each "basis" function
# dof is implicitly calculated based on the first dimension of theta
def drift(x, theta):
    dof = theta.shape[0]
    out = np.dot(mypoly(x, dof), theta)
    return out

# create sample paths! 

# this function creates a bunch of Euler-Maruyama paths from an array
# of initial conditions

# coef = coefficients of drift function in "mypoly" basis
# g = diffusion coefficient
# numsteps = total number of interal time steps to take
# h = size of each internal time step
# savesteps = number of times to save the solution
# ic = vector of initial conditions
# it = corresponding vector of initial times

def createpaths(coef, g, numsteps, savesteps, h, ic, it):
    h12 = np.sqrt(h)
    numpaths = ic.shape[0]
    x = np.zeros(( (savesteps + 1), numpaths))
    t = np.zeros(( (savesteps + 1), numpaths))
    x[0, :] = ic
    t[0, :] = it
    curx = x[0, :].copy()
    curt = t[0, :].copy()
    j = 1
    for i in range(1, numsteps + 1):
        curx += drift(curx, coef)*h + g*h12*np.random.standard_normal(numpaths)
        curt += h
        if (i % (numsteps // savesteps) == 0):
            x[j, :] = curx
            t[j, :] = curt
            j += 1

    return x, t

def brownianbridge(g, xin, tin, n, index):
    h = (tin[index + 1] - tin[index]) / n
    tvec = tin[index] + (1+np.arange(n))*h
    h12 = np.sqrt(h)
    wincs = np.random.normal(scale=h12*g, size=n)
    w = np.cumsum(wincs)
    bridge = xin[index] + w - ((tvec - tin[index])/(tin[index + 1]-tin[index]))*(w[n-1] + xin[index] - xin[index + 1])
    tvec = np.concatenate((tin[[index]], tvec))
    bridge = np.concatenate((xin[[index]],bridge))
    return tvec, bridge

def girsanov(g, path, dt, theta):
    b = drift(path, theta)
    int1 = np.dot(b[:-1]/(g*g), np.diff(path))
    b2 = np.square(b)/(g*g)
    int2 = np.sum(0.5*(b2[1:] + b2[:-1]))*dt
    r = int1 - 0.5*int2
    return r

