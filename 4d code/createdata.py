import numpy as np
import data_creation as dc
import parameters as prm

"""
Sample equation is the coupled Pendulum
theta_1'' = -g/L sin(theta_1) - k/m(theta_1 - theta_2)
theta_2'' = -g/L sin(theta_2) + k/m(theta_1 - theta_2)

Using the small angle approximation or Maclaurin expansion of 
the sin term, the equations can be simplified to:
theta_1'' = -g/L (theta_1 - theta_1^3 / 6) - k/m(theta_1 - theta_2)
theta_2'' = -g/L (theta_2 - theta_2^3 / 6) - k/m(theta_1 - theta_2)

Converting this second order system of equations to a first order system of equations:
x_1' = x_2 dt + g_1 dW_1
x_2' = [-g/L (x_1 - x_1^3 / 6) - k/m(x_1 - x_3)]dt + g_2 dW_2
x_3' = x_4 dt + g_3 dW_3
x_4' = [-g/L (x_3 - x_3^3 / 6) - k/m(x_1 - x_3)]dt + g_2 dW_2

Standard parameter values:
"""
sim_param = prm.system()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[1., 0., 3.2, 0.], [0.5, 0., 0.2, 0.], [-1.2, 0., 0.34, 0.], [0.98, 0., -1.34, 0.], [0.5, 0., -1.5, 0.], [0.1, 0., 0.9, 0.]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('nem_4D.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)
