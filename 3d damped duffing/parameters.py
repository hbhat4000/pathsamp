import numpy as np

def find_dof(degree):
	if (degree <= 1):
		return 1

	return int(degree * (degree + 1) / 2) + find_dof(degree - 1)

polynomial_degree = 4
dim = 3
dof = find_dof(polynomial_degree)

class em:
	def __init__(self, tol, burninpaths, mcmcpaths, numsubintervals, niter, dt):
		self.tol = tol	# tolerance for error in the theta value
		self.burninpaths = burninpaths 	# burnin paths for mcmc
		self.mcmcpaths = mcmcpaths	# sampled paths for mcmc
		self.numsubintervals = numsubintervals	# number of sub intervals in each interval [x_i, x_{i+1}] for the Brownian bridge
		self.niter = niter	# threshold for number of EM iterations, after which EM returns unsuccessfully
		self.h = dt / numsubintervals	# time step for EM

	def __init__(self, dt):
		self.tol = 1e-3
		self.burninpaths = 50
		self.mcmcpaths = 500
		self.numsubintervals = 10
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
		self.ic = ic
		self.it = it
		self.numpaths = ic.shape[0]

class system:
	def __init__(self, alpha, beta, gamma, delta, omega, gvec):
		self.alpha = alpha
		self.beta = beta
		self.gamma = gamma
		self.delta = delta
		self.omega = omega
		self.gvec = gvec

	def __init__(self):
		self.alpha = -1.
		self.beta = 1.
		self.gamma = 0.5 # varying between 0.20 to 0.65
		self.delta = 0.3
		self.omega = 1.2
		self.gvec = np.array([0.01, 0.01, 0.01])