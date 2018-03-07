import numpy as np

# these parameters define the model
# true_theta defines the basis coefficients in thedrift function
polynomial_degree = 4
dim = 2

# last term is dof choose dim_x = (dof!) / ((dof - dim)! (dim)!)
dof = 1 + (polynomial_degree - 1) * (dim) + polynomial_degree * (polynomial_degree - 1)

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
		self.numsubintervals = 10
		self.niter = 100
		self.h = dt / self.numsubintervals

class data:
	def __init__(self, theta, gvec):
		self.theta = theta
		self.gvec = gvec

class euler_maruyama:
	def __init__(self, numsteps, savesteps, ft, ic, it):
		self.numsteps = numsteps;
		self.savesteps = savesteps;
		self.ft = ft;
		self.h = ft / numsteps;
		self.ic = ic;
		self.it = it;

	def __init__(self):
		self.numsteps = 25000	# number of intermediate Euler Maruyama steps
		self.savesteps = 100	# number of time steps saved in the synthetic data
		self.ft = 10.0			# final time step
		self.h = self.ft / self.numsteps	# Euler Maruyama time step
		self.ic = np.array([[1., 1.], [0., 0.], [-1., -1.]])	# 2D array of initial conditions for x
		self.it = np.zeros(self.ic.shape)	# 2D array of initial time points for t