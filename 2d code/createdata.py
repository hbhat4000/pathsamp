import numpy as np
import newbridge as nb

# these parameters define the model
# true_theta defines the basis coefficients in thedrift function
polynomial_degree = 4
dim = 2

# last term is dof choose dim_x = (dof!) / ((dof - dim)! (dim)!)
dof = 1 + (polynomial_degree - 1) * (dim) + polynomial_degree * (polynomial_degree - 1)
true_theta = np.zeros((dof, dim))

# sample equation is the Duffing oscillator
# dx = y dt + g_1 dWt
# dy = (-x -x^3)dt + g_2 dWt
true_theta[4, 0] = 1
true_theta[1, 1] = -1
true_theta[3, 1] = -1

# g is the diffusion coefficient
true_g = np.array([0.25, 0.25])

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
ic = np.array([[1., 1.], [0., 0.], [-1., -1.]])

# 2D array of initial time points for t
it = np.zeros(ic.shape)

# create paths
xout, tout = nb.createpaths(true_theta, true_g, numsteps, savesteps, h, ic, it)

# save to file
import pickle
with open('nem_2D.pkl','wb') as f:
    pickle.dump([xout, tout, true_theta, true_g], f)