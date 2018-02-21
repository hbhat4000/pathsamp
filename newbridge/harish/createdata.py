import numpy as np
import newbridge as nb

# these parameters define the model
# coef essentially defines the drift function
mycoef = np.array([1., 2., -1., -0.5])
# g is the diffusion coefficient
myg = 1/2

# these are timestepping parameters
numsteps = 25000
savesteps = 100
ft = 10.0
h = ft/numsteps
ic = np.array([1.0, -1.0])
it = np.array([0.0, 0.0])

# create paths
xout, tout = nb.createpaths(mycoef, myg, numsteps, savesteps, h, ic, it)

# save to file
import pickle
with open('nem.pkl','wb') as f:
    pickle.dump([xout,tout], f)

