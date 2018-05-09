import numpy as np
import data_creation as dc
import parameters as prm

sim_param = prm.system()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[1.], [1.2], [0.8], [0.6], [0.4], [2.5], [3.], [-0.25], [-0.5], [-0.6]])
# ic = np.random.randn(10, 1) + 2
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('./data/nem_1D_noise1.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)
