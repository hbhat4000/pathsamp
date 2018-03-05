import numpy as np
import newbridge as nb

# these parameters define the model
# coef essentially defines the drift function
true_theta = np.array([1., 2., -1., -0.5])

# g is the diffusion coefficient
true_g = 0.5

# these are timestepping parameters
# number of intermediate Euler Maruyama steps
numsteps = 25000

# number of time steps saved in the synthetic data
savesteps = 100

# final time step
ft = 10.0

# Euler Maruyama time step
h = ft/numsteps

# array of initial conditions for x
ic = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])

# array of initial time points for t
it = np.zeros(ic.shape[0])

# create paths
xout, tout = nb.createpaths(true_theta, true_g, numsteps, savesteps, h, ic, it)

# save to file
import pickle
with open('nem.pkl','wb') as f:
    pickle.dump([xout,tout], f)