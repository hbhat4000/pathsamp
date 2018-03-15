import numpy as np
import newbridge as nb
import parameters as prm

"""
Sample equation is the Duffing oscillator
dx = y dt + g_1 dWt
dy = (-x -x^3)dt + g_2 dWt

dimension = 2
degree of freedom for 3rd order hermite polynomial = 19
"""
theta = np.zeros((prm.dof, prm.dim))
gvec = np.array([0.25, 0.25])

d_param = prm.data(theta, gvec)
d_param.theta[4, 0] = 1
d_param.theta[1, 1] = -1
d_param.theta[3, 1] = -1

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[1., 1.], [0.8, 0.8], [0.4, 0.4], [0.2, 0.2]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout = nb.createpaths(d_param, euler_param)

# save to file
import pickle
with open('nem_2D.pkl','wb') as f:
    pickle.dump([xout, tout, d_param, euler_param], f)