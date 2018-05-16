import numpy as np
import newbridge as nb
import parameters as prm
import data_creation as dc
import pickle
import os
import random

# for running the code using a job array on cluster
parvalue = int(os.environ['SGE_TASK_ID']) * 10 + 1

# load data, noise_1 is data with noise = 0.1
with open('./data/noise_1.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

numpaths = allx.shape[0]
numpoints = parvalue - 2

sorted_indices = np.zeros((numpaths, parvalue))
for i in range(numpaths):
    random_indices = random.sample(range(0, allx.shape[1]), numpoints)
    sorted_indices[i, 0] = 0
    sorted_indices[i, numpoints + 1] = allx.shape[1] - 1
    sorted_indices[i, 1:(numpoints+1)] = np.sort(random_indices)

new_indices = sorted_indices.astype(int)

for i in range(numpaths):
    x = allx[:, new_indices[i, :], :]
    t = allt[:, new_indices[i, :]]

data_param = prm.data(theta = 0.5 * np.random.rand(prm.dof, prm.dim), gvec = sim_param.gvec)

print("Data shape:", x.shape)
print("Theta shape:", data_param.theta.shape)
print("Theta:", data_param.theta)

em_param = prm.em(tol=1e-2, burninpaths=10, mcmcpaths=100, numsubintervals=5, niter=100, dt=(allt[0, 1] - allt[0, 0]))

# call to EM which returns the final error and estimated theta value
error_list, theta_list = nb.em(x, t, em_param, data_param)

estimated_theta = prm.theta_transformations(theta=theta_list[-1], theta_type='hermite')
true_theta = prm.theta_transformations(theta=dc.true_theta(sim_param), theta_type='ordinary')

print("\n Estimated ordinary: ", np.transpose(estimated_theta.ordinary), "\n Estimated hermite: ", np.transpose(estimated_theta.hermite), "\n Estimated ordinary sparse: ", np.transpose(estimated_theta.sparse_ordinary), "\n Estimated hermite sparse: ", np.transpose(estimated_theta.sparse_hermite))

print("True ordinary: ", np.transpose(true_theta.ordinary), "\n True hermite: ", np.transpose(true_theta.hermite))

errors = nb.norm_error(true_theta, estimated_theta)
print("Ordinary error: ", errors[0], ", Hermite error: ", errors[1], ", Sparse ordinary error: ", errors[2], ", Sparse hermite error: ", errors[3])

estimated_param = prm.data(theta = estimated_theta.hermite, gvec = sim_param.gvec)
inferred_gvec = nb.infer_noise(x, t, em_param, estimated_param)
print("Inferred gvec: ", inferred_gvec)
errors.append(inferred_gvec - sim_param.gvec)

print("\n")

# save to file
with open('./random_timepoints/rand_' + str(parvalue) + '.pkl','wb') as f:
    pickle.dump([error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param], f)

