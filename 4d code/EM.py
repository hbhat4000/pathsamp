import numpy as np
import newbridge as nb
import parameters as prm

# load data
import pickle
with open('./varying_subintervals/data/common_data.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

data_param = prm.data(theta = 0.5 * np.random.rand(prm.dof, prm.dim), gvec = sim_param.gvec)

print("Data shape:", allx.shape)
print("Theta shape:", data_param.theta.shape)
print("Theta:", data_param.theta)

"""
Default parameters for Expectation-Maximization
em_param = param.em(tol = 1e-2, burninpaths = 10, mcmcpaths = 100, numsubintervals = 10, niter = 100, dt = (allt[0, 1] - allt[0, 0]))
"""
em_param = prm.em(dt = allt[0, 1] - allt[0, 0])

# call to EM which returns the final error and estimated theta value
error_list, theta_list = nb.em(allx, allt, em_param, data_param)

# convert the hermite polynomial to simplified polynomial expression
transformed_theta = nb.hermite_to_ordinary(theta_list[-1])

print("Error: ", error_list)
print("Theta: ", theta_list)

print("Transformed theta without sparsity: ", transformed_theta)
transformed_theta[np.abs(transformed_theta) < 0.1] = 0.
print("Transformed theta with sparsity: ", transformed_theta)

# save to file
with open('./varying_subintervals/results/subint_10.pkl','wb') as f:
    pickle.dump([error_list, theta_list, transformed_theta, em_param, data_param], f)
