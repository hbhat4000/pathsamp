import numpy as np
import pickle
from error_plots import error_plots as ep
from tables import error_tables as et
from matplotlib import pyplot as plt
import parameters as prm

# 1) Data used
meta_error_list = []
for i in range(1, 11):
    with open('./random_timepoints/rand_' + str(i*10+1) + '.pkl', 'rb') as f:
        x, t, error_list, theta_list, gammavec_list, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors))

    fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    titles = [r'$x_0$', r'$x_1$', r'$x_2$']

    for j in range(10):
        y_vals = [x[j, :, 0], x[j, :, 1], x[j, :, 2]]
        for ax, title, y in zip(axes.flat, titles, y_vals):
            ic = ((np.round(euler_param.ic[i], 2)))
            ax.plot(t[j, :], y, label='ic:'+str(ic))
            ax.set_title(title)
            ax.grid(True)

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.suptitle('Observed data used for random time point experiments in 3Dl, number of time steps = ' + str(i*10+1))
    plt.savefig('./random_timepoints/plots/rand_' + str(i*10+1) + '.pdf', format = 'pdf', bbox_inches='tight')
    plt.close()

################################################################################################

# 2) Error plots
parval = meta_error_list[0][0][0]
tp_mapping = []

for i in range(parval):
    tp_mapping.append(int(meta_error_list[i][0][1]))

hermite_errors = np.zeros((threshold.shape[0], 6, parval))
ordinary_errors = np.zeros((threshold.shape[0], 6, parval))
estimated_theta = np.zeros((parval, prm.dof, prm.dim))
true_theta = np.zeros((prm.dof, prm.dim))

for val in range(parval):
    for th in range(threshold.shape[0]):
        for fn in range(6):
            hermite_errors[th][fn][val] = meta_error_list[val][5][th][fn]
            ordinary_errors[th][fn][val] = meta_error_list[val][4][th][fn]
    estimated_theta[val] = meta_error_list[val][1].hermite
true_theta = meta_error_list[0][2].hermite

exp = 'random_timepoints'
threshold = meta_error_list[0][3]

ep(exp, hermite_errors, ordinary_errors, parval, tp_mapping, threshold)
et(exp, hermite_errors, ordinary_errors, estimated_theta, true_theta, parval, tp_mapping, threshold)

###################################################################################################

# 3) Comparison of true drift function vs estimated drift function
def f(theta, x):
    y = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        y[i, :] = index(theta[:, i], x)
    return y  

def index(theta, x):
    y = np.zeros((x.shape[1]))
    index = 0

    for d in range(0, prm.num_hermite_terms):
        for k in range(0, d + 1):
            for j in range(0, d + 1):
                for i in range(0, d + 1):
                    if (i + j + k == d):
                        y += theta[index] * np.power(x[0, :], i) * np.power(x[1, :], j) * np.power(x[2, :], k)
                        index += 1

    return y

x_sparse = np.arange(-2.0, 2.0, 0.5)
x_dense = np.arange(-2.0, 2.0, 0.1)
x_true = np.array((x_sparse, x_sparse, x_sparse))
x_est = np.array((x_dense, x_dense, x_dense))

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True)
fig.set_figwidth(20)
fig.set_figheight(5)
titles = [r'$x_0$', r'$x_1$', r'$x_2$']

for i in range(parval):
    f_true = f(np.array(meta_error_list[i][2].ordinary), x_true)
    f_est = f(np.array(meta_error_list[i][1].ordinary), x_est)
    y_vals_true = [f_true[0, :], f_true[1, :], f_true[2, :]]
    y_vals_est = [f_est[0, :], f_est[1, :], f_est[2, :]]

    for ax, title, y_true, y_est in zip(axes.flat, titles, y_vals_true, y_vals_est):
        if (i == 0):
            ax.plot(x_true[0, :], y_true, 'bo', label='true drift')
        ax.plot(x_est[0, :], y_est, label='time points = '+str(meta_error_list[i][0][1]))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-2., 3., 1.))
        ax.set_yticks(np.arange(-5.0, 5.0, 1.))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-5, 5])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions in 3Dl')
plt.savefig('./random_timepoints/plots/drift_comparison.pdf', format = 'pdf', bbox_inches='tight')
plt.close()
