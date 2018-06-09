import numpy as np
import polynomial_functions as pfn
import scipy.special

def find_dof(degree, dim):
	return int(scipy.special.binom(degree + dim - 1, dim))

num_hermite_terms = 4
dim = 6
dof = find_dof(num_hermite_terms, dim)

class em:
    def __init__(self, tol, burninpaths, mcmcpaths, numsubintervals, niter, dt):
        self.tol = tol  # tolerance for error in the theta value
        self.burninpaths = burninpaths  # burnin paths for mcmc
        self.mcmcpaths = mcmcpaths      # sampled paths for mcmc
        self.numsubintervals = numsubintervals  # number of sub intervals in each interval [x_i, x_{i+1}] for the Brownian bridge
        self.niter = niter      # threshold for number of EM iterations, after which EM returns unsuccessfully
        self.h = dt / numsubintervals   # time step for EM

class data:
    def __init__(self, theta, gvec):
        self.theta = theta
        self.gvec = gvec

class euler_maruyama:
    def __init__(self, numsteps, savesteps, ft, ic, it, numpaths):
        self.numsteps = numsteps
        self.savesteps = savesteps
        self.ft = ft
        self.h = ft / numsteps
        self.ic = ic
        self.it = it
        self.numpaths = numpaths

class system:
    def __init__(self, kvec, mvec, gvec):
        self.kvec = kvec
        self.mvec = mvec
        self.gvec = gvec

class theta_transformations:
    def __init__(self, theta, theta_type=None):
        if theta_type is 'ordinary':
            self.ordinary = theta
            self.hermite = pfn.ordinary_to_hermite(theta)
        if theta_type is 'hermite':
            self.ordinary = pfn.hermite_to_ordinary(theta)
            self.hermite = theta
