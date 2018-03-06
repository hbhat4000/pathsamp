import numpy as np
import newbridge as nb

# load data
import pickle
with open('nem.pkl','rb') as f:
    allx, allt = pickle.load(f)

# to check if x and t array are of correct shapes
print(allx.shape)
print(allt.shape)

# initial guess for theta
theta = np.array([2., 1., 0., -0.1])

# degree of freedom is the highest degree polynomial in
# the hermite series, which is the number of theta parameters
dof = theta.shape[0]

# true diffusion coefficient
g = 0.5

# number of sub intervals in each interval [x_i, x_{i+1}] for
# the Brownian bridge
numsubintervals = 20

# time step for EM
h = (allt[1,0] - allt[0,0])/numsubintervals

# number of Brownian bridge sample paths sampled
numpaths = 1000

# number of Brownian bridge 
burninpaths = 100

# tolerance for error in the theta value
mytol = 1e-3

# threshold for number of EM iterations, after which EM returns unsuccessfully
niter = 100

# call to EM which returns the final error and estimated theta value
error, theta = nb.em(mytol, niter, burninpaths, numpaths, g, allx, allt, numsubintervals, h, theta, dof)
print("Error", error)
print("Theta: ", theta)