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

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    titles = [r'$x_0$', r'$x_1$']

    for i in range(10):
        y_vals = [x[i, :, 0], x[i, :, 1]]
        for ax, title, y in zip(axes.flat, titles, y_vals):
            ic = ((np.round(euler_param.ic[i], 2)))
            ax.plot(t[i, :], y, label='ic:'+str(ic))
            ax.set_title(title)
            ax.grid(True)
            ax.set_xticks(np.arange(0, 11, 1))
            ax.set_yticks(np.arange(-2.0, 2.0, 0.5))
            ax.set_xlim([0, 10])
            ax.set_ylim([-2, 2])

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.suptitle('Observed data used for experiments in 2D, noise = ' + str(noise_mapping[parvalue]))
    plt.savefig('./data/plots/noise_' + str(parvalue) + '.pdf', format = 'pdf', bbox_inches='tight')
    plt.close()
