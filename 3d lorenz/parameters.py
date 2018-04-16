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
		self.burninpaths = 10
		self.mcmcpaths = 100
		self.numsubintervals = 20
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
		self.numsteps = 25000 * 40
		self.savesteps = 400
		self.ft = 40.0
		self.h = self.ft / self.numsteps
		self.ic = ic
		self.it = it
		self.numpaths = ic.shape[0]

class lorenz:
	def __init__(self, sigma, rho, beta, gvec):
		self.sigma = sigma
		self.rho = rho
		self.beta = beta
		self.gvec = gvec

	def __init__(self):
		self.sigma = 10.0
		self.rho = 28.0
		self.beta = 8.0 / 3.0
		self.gvec = np.array([1e-6, 1e-6, 1e-6])