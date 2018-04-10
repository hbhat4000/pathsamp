import numpy as np
import data_creation as dc
import parameters as prm

"""
Sample equation is the Lorenz oscillator

dx = sigma(y - x) dt + g_1 dWt
dy = (x(rho - z) - y) dt + g_2 dWt
dz = (xy - beta z) dt + g_3 dWt

sigma = 10, rho = 28, beta = 8/3
g_1 = 1e-2, g_2 = 1e-2, g_3 = 1e-2
"""
sim_param = prm.lorenz()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[8.2, 4.1, 1.3], [5.3, 2.2, 0.9], [7.2, -1.8, 2.1], [-0.9, -1.4, 0.4], [0.6, 0.9, -1.8], [-6., 2.3, 4.5], [-0.2, -2.9, 1.4], [3.5, 2.6, -0.5], [0.2, 0., 1.2], [-1.5, 0.6, 3.4], [6.7, 2.2, 4.5]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('3D_lorenz.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)