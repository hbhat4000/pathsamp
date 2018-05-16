import numpy as np
import data_creation as dc
import newbridge as nb

def choose(degree, dim):
	return np.math.factorial(degree) / (np.math.factorial(dim) * np.math.factorial(degree - dim))

def find_dof(degree, dim):
	return int(choose(degree + dim - 1, dim))

num_hermite_terms = 4
dim = 1
dof = find_dof(num_hermite_terms, dim) 

class em:
	def __init__(self, tol, burninpaths, mcmcpaths, numsubintervals, niter, dt):
		self.tol = tol	# tolerance for error in the theta value
		self.burninpaths = burninpaths 	# burnin paths for mcmc
		self.mcmcpaths = mcmcpaths	# sampled paths for mcmc
		self.numsubintervals = numsubintervals	# number of sub intervals in each interval [x_i, x_{i+1}] for the Brownian bridge
		self.niter = niter	# threshold for number of EM iterations, after which EM returns unsuccessfully
		self.h = dt / numsubintervals	# time step for EM

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
	def __init__(self, alpha, beta, gamma, gvec):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.gvec = gvec

class theta_transformations:
	def __init__(self, theta, theta_type=None):
		if theta_type is 'ordinary':
			self.ordinary = theta
			self.hermite = nb.ordinary_to_hermite(theta)
			self.sparse_ordinary = nb.theta_sparsity(self.ordinary)
			self.sparse_hermite = nb.theta_sparsity(self.hermite)
		if theta_type is 'hermite':
			self.ordinary = nb.hermite_to_ordinary(theta)
			self.hermite = theta
			self.sparse_ordinary = nb.theta_sparsity(self.ordinary)
			self.sparse_hermite = nb.theta_sparsity(self.hermite)
