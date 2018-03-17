import numpy as np
import operator as op

# these parameters define the model
# true_theta defines the basis coefficients in thedrift function
polynomial_degree = 3
dim = 3
dof = 1 + dim * (polynomial_degree - 1) + dim * np.power((polynomial_degree - 1), 2) + np.power(polynomial_degree - 1, dim)

# dof = 64 	# for dim = 3, polynomial_degree = 4
# dof = 27 	# for dim = 3, polynomial_degree = 3
# dof = 8  	# for dim = 3, polynomial_degree = 2

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
		self.mcmcpaths = 50
		self.numsubintervals = 10
		self.niter = 100
		self.h = dt / self.numsubintervals

class data:
	def __init__(self, theta, gvec):
		self.theta = theta
		self.gvec = gvec

class euler_maruyama:
	def __init__(self, numsteps, savesteps, ft, ic, it, numpaths):
		self.numsteps = numsteps;
		self.savesteps = savesteps;
		self.ft = ft;
		self.h = ft / numsteps;
		self.ic = ic;
		self.it = it;
		self.numpaths = numpaths;

	def __init__(self, ic, it):
		self.numsteps = 25000
		self.savesteps = 100
		self.ft = 10.0
		self.h = self.ft / self.numsteps
		self.ic = ic
		self.it = it
		self.numpaths = ic.shape[0]