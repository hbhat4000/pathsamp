import numpy as np
import data_creation as dc
import parameters as prm

"""
Sample equation is the coupled system of springs
m_0 theta_0'' = -k_0 (theta_0) - k_1(theta_0 - theta_1)
m_1 theta_1'' = -k_1 (theta_1 - theta_0) - k_2 (theta_1 - theta_2)
m_2 theta_2'' = -k_2 (theta_2 - theta_1) - k_3 (theta_2)

Converting this second order system of equations to a first order system of equations:
x_0' = x_1 dt + g_0 dW_0
x_1' = [-k_0/m_0 (x_0) - k_1/m_0 (x_0 - x_2)]dt + g_1 dW_1
x_2' = x_3 dt + g_2 dW_2
x_3' = [-k_1/m_1 (x_2 - x_0) - k_2/m_1 (x_2 - x_4)]dt + g_3 dW_3
x_4' = x_5 dt + g_4 dW_4
x_5' = [-k_2/m_2 (x_2 - x_0) - k_3/m_2 (x_2)]dt + g_5 dW_5
"""
sim_param = prm.system()

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.random.randn(10, 6)
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('./varying_subintervals/data/common_data.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)
