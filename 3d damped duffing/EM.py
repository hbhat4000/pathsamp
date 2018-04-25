import numpy as np
import newbridge as nb
import parameters as prm

# load data
import pickle
with open('nem_3D.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

"""
dimension = 3
degree of freedom for 3rd order hermite polynomial = 64
"""
data_param = prm.data(theta = 0.5 * np.random.rand(prm.dof, prm.dim), gvec = sim_param.gvec)
print("Data shape:", allx.shape)
print("Theta shape:", data_param.theta.shape)
print("Theta:", data_param.theta)

"""
Default parameters for Expectation-Maximization
em_param = param.em(tol = 1e-3, burninpaths = 10, mcmcpaths = 100, numsubintervals = 10, niter = 100, dt = (allt[0, 1, 0] - allt[0, 0, 0]))
"""
em_param = prm.em(dt = allt[0, 1] - allt[0, 0])

# call to EM which returns the final error and estimated theta value
error, d_param = nb.em(allx, allt, em_param, data_param)

print("Error", error)
print("Theta: ", d_param.theta)
print("gvec:", d_param.gvec)