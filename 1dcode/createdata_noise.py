import numpy as np
import data_creation as dc
import parameters as prm
import os

# for running the code using a job array on cluster
# parvalue is between 1 and 8
parvalue = int(os.environ['SGE_TASK_ID']) - 1

noise_mapping = (0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001)
sim_param = prm.system(alpha = 1., beta = 1., gamma = -1., gvec = np.full(prm.dim, noise_mapping[parvalue]))

# create paths
numpaths = 10
ic = np.random.uniform(low=-0.6, high=3.0, size=(numpaths, prm.dim))
it = np.zeros((numpaths))
euler_param = prm.euler_maruyama(numsteps=100000, savesteps=1000, ft=10., ic=ic, it=it, numpaths=numpaths)
xout, tout, xout_without_noise = dc.createpaths(euler_param, sim_param)

# save to file
import pickle
with open('./data/noise_' + str(parvalue) + '.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, euler_param, sim_param], f)
