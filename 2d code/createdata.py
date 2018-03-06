import numpy as np
import newbridge as nb

# these parameters define the model
# true_theta defines the basis coefficients in thedrift function
dof = 4
dim_x = 2

# last term is dof choose dim_x = (dof!) / ((dof - dim_x)! (dim_x)!)
theta_space = 1 + (dof - 1) * (dim_x) + dof * (dof - 1)
true_theta = np.array((theta_space, dim_x))

# sample equation is the Duffing oscillator
# dx = y dt + g_1 dWt
# dy = (-x -x^3)dt + g_2 dWt
true_theta[0, 4] = 1
true_theta[1, 1] = -1
true_theta[1, 3] = -1

# g is the diffusion coefficient
true_g = np.array([0.5, 0.5])

# these are timestepping parameters
# number of intermediate Euler Maruyama steps
numsteps = 25000

# number of time steps saved in the synthetic data
savesteps = 100

# final time step
ft = 10.0

# Euler Maruyama time step
h = ft/numsteps

# 2D array of initial conditions for x
ic = np.array([1., 1.], [0., 0.], [-1., -1.])

# 2D array of initial time points for t
it = np.zeros(ic.shape)

# create paths
xout, tout = nb.createpaths(true_theta, true_g, numsteps, savesteps, h, ic, it)

# save to file
import pickle
with open('nem_2D.pkl','wb') as f:
    pickle.dump([xout,tout], f)