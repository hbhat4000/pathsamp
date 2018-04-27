import numpy as np
import data_creation as dc
import parameters as prm

"""
Sample equation is the 2N coupled masses with spring
with weights w1, ..., wN and spring constants k0, ..., kN:
m_(2n) x''_(2n) = -k_(2n) (x_(2n) - x_(2n-1)) - k_(2n+1) (x_2n - x_(2n+1))
[Reference: http://www.people.fas.harvard.edu/~djmorin/waves/normalmodes.pdf]

Converting this second order system of equations to a first order system of equations:
dx_n = x_(n+1)

Standard parameter values:
"""
sim_param = prm.system()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
#ic = np.array([[1., 0., 3.2, 0.], [0.5, 0., 0.2, 0.], [-1.2, 0., 0.34, 0.], [0.98, 0., -1.34, 0.], [0.5, 0., -1.5, 0.], [0.1, 0., 0.9, 0.]])
ic = np.random.random((6, 10))
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('nem_6D.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)
