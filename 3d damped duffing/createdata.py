import numpy as np
import data_creation as dc
import parameters as prm

"""
Sample equation is the damped Duffing oscillator
x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t) = gamma (1 - 0.5(omega t)^2)
There can be multiple ways of representing the t-z substitution: z= t, z = omega t, z = cos(omega t)
The first order system of equations are:
dx = y dt + g_1 dWt
dy = (alpha x - beta x^3 - delta y + gamma (1 - 0.5 z^2)) dt + g_2 dWt
dz = omega dt + g_3 dWt

Standard parameter values:
alpha = -1, beta = +1, gamma = 0.2 to 0.65, delta = 0.3, omega = 1.2
g_1 = 1e-2, g_2 = 1e-2, g_3 = 1e-2
"""
sim_param = prm.system()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[2., -1.3, 0.], [1.2, 1.7, 0.], [1., 0., 0.], [0.6, 0.9, 0.], [0.1, 2., 0.], [0.8, 0.5, 0.], [0.4, 0.2, 0.], [-0.25, 0.1, 0.], [1.0, 0.5, 0.], [-2., -1., 0.], [-3., -2.3, 0.], [-1., -0.23, 0.], [-0.5, 2.,0.]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('nem_3D.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)