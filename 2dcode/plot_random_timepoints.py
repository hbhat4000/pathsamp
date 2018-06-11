import numpy as np
import pickle
from error_plots import error_plots as ep
from matplotlib import pyplot as plt
import parameters as prm

# 1) Data used
meta_error_list = []
for i in range(1, 11):
    with open('./random_timepoints/rand_' + str(i*10+1) + '.pkl', 'rb') as f:
        x, t, error_list, theta_list, gammavec_list, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors))

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(-1.0, 3.0, 0.5))

    for j in range(10):
        plt.plot(t[j, :], x[j, :, 0], label='initial condition ' + str(euler_param.ic[j]))

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.title('Observed data used for random time point experiments, number of time steps = ' + str(i*10+1))
    plt.grid()
    plt.savefig('./random_timepoints/plots/rand_' + str(i*10+1) + '.eps', format = 'eps', bbox_inches='tight')

################################################################################################

# 2) Error plots
parval = meta_error_list[0][0][0]
tp_mapping = []

for i in range(parval):
    tp_mapping.append(int(meta_error_list[i][0][1]))

hermite_errors = np.zeros((threshold.shape[0], 6, parval))
ordinary_errors = np.zeros((threshold.shape[0], 6, parval))

for th in range(threshold.shape[0]):
    for fn in range(6):
        for val in range(parval):
            hermite_errors[th][fn][val] = meta_error_list[val][5][th][fn]
            ordinary_errors[th][fn][val] = meta_error_list[val][4][th][fn]

exp = 'random_timepoints'
threshold = meta_error_list[0][3]

ep(exp, hermite_errors, ordinary_errors, parval, tp_mapping, threshold)

################################################################################################

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
        for j in range(0, d + 1):
            for i in range(0, d + 1):
                if (i + j == d):
                    y += theta[index] * np.power(x[0, :], i) * np.power(x[1, :], j)
                    index += 1

    return y

x_sparse = np.arange(-2.0, 2.0, 0.5)
x_dense = np.arange(-2.0, 2.0, 0.1)
x_true = np.array((x_sparse, x_sparse))
x_est = np.array((x_dense, x_dense))

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(5)
titles = [r'$x_0$', r'$x_1$']

for i in range(parval):
    f_true = f(np.array(meta_error_list[i][2].ordinary), x_true)
    f_est = f(np.array(meta_error_list[i][1].ordinary), x_est)
    y_vals_true = [f_true[0, :], f_true[1, :]]
    y_vals_est = [f_est[0, :], f_est[1, :]]

    for ax, title, y_true, y_est in zip(axes.flat, titles, y_vals_true, y_vals_est):
        if (i == 0):
            ax.plot(x_true[0, :], y_true, 'bo', label='true drift')
        ax.plot(x_est[0, :], y_est, label='time points = '+str(meta_error_list[i][0][1]))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-2, 3, 1))
        ax.set_yticks(np.arange(-5.0, 5.0, 1.))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-5, 5])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions')
plt.savefig('./random_timepoints/plots/drift_comparison.eps', format = 'eps', bbox_inches='tight')

