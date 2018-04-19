import numpy as np
import newbridge as nb
import parameters as prm

# load data
import pickle
with open('nem_1D_noise9.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

data_param = prm.data(theta = 0.5 * np.random.rand(prm.dof, prm.dim), gvec = sim_param.gvec)

print("Data shape:", allx.shape)
print("Theta shape:", data_param.theta.shape)
print("Theta:", data_param.theta)

"""
Default parameters for Expectation-Maximization
em_param = param.em(tol = 1e-3, burninpaths = 100, mcmcpaths = 1000, numsubintervals = 10, niter = 100, dt = (allt[0, 1, 0] - allt[0, 0, 0]))
"""
em_param = prm.em(dt = allt[0, 1] - allt[0, 0])

# call to EM which returns the final error and estimated theta value
error_list, theta_list = nb.em(allx, allt, em_param, data_param)

print("Error", error_list)
print("Theta: ", theta_list)

# save to file
import pickle
with open('1D_result_noise9.pkl','wb') as f:
    pickle.dump([error_list, theta_list, em_param, data_param], f)
