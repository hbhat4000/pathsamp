import numpy as np
from em_functions import em as em
import polynomial_functions as pfn
import parameters as prm
import data_creation as dc
import pickle
import os

# for running the code using a job array on cluster
parvalue = int(os.environ['SGE_TASK_ID']) - 1

# load data
with open('./data/noise_' + str(parvalue) + '.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

# picking parvalue number of time series and the coarseness of the observed data
x = allx[:, 0::10, :] # picking every 10th term to get a total of 101 time points
t = allt[:, 0::10] # 101 time points

data_param = prm.data(theta = 0.5 * np.random.rand(prm.dof, prm.dim), gvec = sim_param.gvec)

print("Data shape:", x.shape)
print("Theta shape:", data_param.theta.shape)
print("Theta:", data_param.theta)

em_param = prm.em(tol=0.1, burninpaths=10, mcmcpaths=100, numsubintervals=5, niter=100, dt=(allt[0, 1] - allt[0, 0]))

# call to EM which returns the final error and estimated theta value
error_list, theta_list, gammavec_list = em(x, t, em_param, data_param)

estimated_theta = prm.theta_transformations(theta=theta_list[-1], theta_type='hermite')
true_theta = prm.theta_transformations(theta=dc.true_theta(sim_param), theta_type='ordinary')

print("\n Estimated ordinary: ", np.transpose(estimated_theta.ordinary), "\n True ordinary: ", np.transpose(true_theta.ordinary))

print("\n Estimated hermite: ", np.transpose(estimated_theta.hermite), "\n True hermite: ", np.transpose(true_theta.hermite))

threshold = np.array([0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
ordinary_errors = []
hermite_errors = []
for th in threshold:
    ordinary_errors.append(pfn.compute_errors(true_theta.ordinary, estimated_theta.ordinary, th))
    hermite_errors.append(pfn.compute_errors(true_theta.hermite, estimated_theta.hermite, th))

print("\n")

# save to file
with open('./varying_noise/noise_' + str(parvalue) + '.pkl','wb') as f:
    pickle.dump([x, t, error_list, theta_list, gammavec_list, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors, em_param, data_param, euler_param, sim_param], f)

