import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

noise_mapping = (0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001)
for parvalue in range(8):
    with open('./data/noise_' + str(parvalue) + '.pkl', 'rb') as f:
       xout, tout, xout_without_noise, euler_param, sim_param = pickle.load(f)

    x = xout[:, 0::10, :]
    t = tout[:, 0::10]

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(-1.0, 3.0, 0.5))

    for i in range(10):
        ic = ((np.round(euler_param.ic[i], 2)))
        plt.plot(t[i, :], x[i, :, 0], label='ic:' + str(ic))

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.title('Observed data used for experiments in 1D, noise = ' + str(noise_mapping[parvalue]))
    plt.grid()
    plt.savefig('./data/plots/noise_' + str(parvalue) + '.pdf', format = 'pdf', bbox_inches='tight')
