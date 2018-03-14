import numpy as np

# these parameters define the model
# true_theta defines the basis coefficients in thedrift function
polynomial_degree = 4
dim = 1

# NOTE : the dof depends on polynomial degree
dof = 1 + (polynomial_degree - 1) * (dim)

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
		self.burninpaths = 100
		self.mcmcpaths = 1000
		self.numsubintervals = 20
		self.niter = 100
		self.h = dt / self.numsubintervals

class data:
	def __init__(self, theta, gvec):
		self.theta = theta
		self.gvec = gvec

class euler_maruyama:
	def __init__(self, numsteps, savesteps, numpaths, ft, ic, it):
		self.numsteps = numsteps;
		self.savesteps = savesteps;
		self.numpaths = numpaths;
		self.ft = ft;
		self.h = ft / numsteps;
		self.ic = ic;
		self.it = it;

	def __init__(self):
		self.numsteps = 25000	# number of intermediate Euler Maruyama steps
		self.savesteps = 100	# number of time steps saved in the synthetic data
		self.ft = 10.0			# final time step
		self.h = self.ft / self.numsteps	# Euler Maruyama time step
		self.ic = np.array([[1.], [0.8], [0.4], [0.2], [-0.2]])	# 2D array of initial conditions for x
		self.numpaths = self.ic.shape[0]		# number of time series paths with different initial conditions
		self.it = np.zeros(self.numpaths)	# 2D array of initial time points for t
