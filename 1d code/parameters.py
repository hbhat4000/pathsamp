import numpy as np

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

	def __init__(self, dt):
		self.tol = 1e-2
		self.burninpaths = 10
		self.mcmcpaths = 100
		self.numsubintervals = 1
		self.niter = 100
		self.h = dt / self.numsubintervals

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

	def __init__(self, ic, it):
		self.numsteps = 25000
		self.savesteps = 100
		self.ft = 10.0
		self.h = self.ft / self.numsteps
		self.ic = np.array([[1.], [1.2], [0.8], [0.6], [0.4], [2.5], [3.], [-0.25], [-0.5], [-0.6]])
		self.it = it
		self.numpaths = ic.shape[0]

class system:
	def __init__(self, alpha, beta, gamma, gvec):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.gvec = gvec

	def __init__(self):
		self.alpha = 1.
		self.beta = 1.
		self.gamma = -1.
		self.gvec = np.array([0.1])
