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

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    fig.set_figwidth(15)
    fig.set_figheight(10)
    titles = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$']

    for i in range(10):
        y_vals = [x[i, :, 0], x[i, :, 1], x[i, :, 2], x[i, :, 3]]
        for ax, title, y in zip(axes.flat, titles, y_vals):
            ic = ((np.round(euler_param.ic[i], 2)))
            ax.plot(t[i, :], y, label='ic:'+str(ic))
            ax.set_title(title)
            ax.grid(True)

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.suptitle('Observed data used for experiments in 4D, noise = ' + str(noise_mapping[parvalue]))
    plt.savefig('./data/plots/noise_' + str(parvalue) + '.pdf', format = 'pdf', bbox_inches='tight')
    plt.close()
