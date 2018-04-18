import numpy as np
import data_creation as dc
import parameters as prm

"""
Sample equation is the Duffing oscillator
dx = (gamma + alpha x + beta x^2) dt + g_1 dWt

Standard parameter values:
alpha = +1, beta = -1, gamma = 1
g_1 = 0.50
"""
sim_param = prm.system()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[1.], [1.2], [0.8], [0.6], [0.4], [2.5], [3.], [1.7], [2.1], [-0.25], [-0.5], [-0.6]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('nem_1D_noise1.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)