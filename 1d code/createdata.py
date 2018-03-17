import numpy as np
import newbridge as nb
import parameters as prm

"""
Sample equation is the Duffing oscillator
dx = (1 + x - x^2) dt + g_1 dWt

dimension = 2
degree of freedom for 3rd order hermite polynomial = 19
"""
theta = np.zeros((prm.dof, prm.dim))
gvec = np.array([0.50])

d_param = prm.data(theta, gvec)
# d_param.theta[0] = 1
d_param.theta[1] = 1
# d_param.theta[2] = -1

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it)
"""
euler_param = prm.euler_maruyama()
xout, tout = nb.createpaths(d_param, euler_param)

# save to file
import pickle
with open('nem_1D.pkl','wb') as f:
    pickle.dump([xout, tout, d_param, euler_param], f)