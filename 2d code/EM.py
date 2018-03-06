import numpy as np
import newbridge as nb

# load data
import pickle
with open('nem_2D.pkl','rb') as f:
    allx, allt, true_theta, true_g = pickle.load(f)

# to check if x and t array are of correct shapes
print(allx.shape)
print(allt.shape)

# initial guess for theta
dof = true_theta.shape[0]
dim = true_theta.shape[1]
theta = np.zeros((dof, dim))

# call to EM which returns the final error and estimated theta value
class em_parameters:
	def __init__(self, mytol, burninpaths, numpaths, numsubintervals, niter, dt):
		self.mytol = mytol	# tolerance for error in the theta value
		self.burninpaths = burninpaths 	# burnin paths for mcmc
		self.numpaths = numpaths	# sampled paths for mcmc
		self.numsubintervals = numsubintervals	# number of sub intervals in each interval [x_i, x_{i+1}] for the Brownian bridge
		self.niter = niter	# threshold for number of EM iterations, after which EM returns unsuccessfully
		self.h = dt / numsubintervals	# time step for EM

class data_parameters:
	def __init__(self, g, theta):
		self.g = g
		self.theta = theta
		self.dof = theta.shape[0]
		self.dim = theta.shape[1]

em_param = em_parameters(1e-3, 10, 100, 20, 100, (allt[0, 1, 0] - allt[0, 0, 0]))
data_param = data_parameters(true_g, theta)

error, theta = nb.em(allx, allt, em_param, data_param)

print("Error", error)
print("Theta: ", theta)