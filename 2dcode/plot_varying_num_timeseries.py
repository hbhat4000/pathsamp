import numpy as np
import pickle
from matplotlib import pyplot as plt
import newbridge as nb
import parameters as prm
from error_plots import error_plots as ep

# 1) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./varying_num_timeseries/ts_' + str(i) + '.pkl','rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param))

parval = 10
ts_mapping = []
for i in range(parval):
    ts_mapping.append(int(meta_error_list[i][0][0]))

exp = 'varying_num_timeseries'
threshold = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
ep(exp, meta_error_list, parval, ts_mapping, threshold)

###################################################################################################

# 2) Comparison of true drift function vs estimated drift function
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
x1 = np.array((x_sparse, x_sparse))
x2 = np.array((x_dense, x_dense))

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(5)
titles = [r'$x_0$', r'$x_1$']

for i in range(parval):
    f2 = f(np.array(meta_error_list[i][3].ordinary), x2)
    f1 = f(np.array(meta_error_list[i][4].ordinary), x1)
    y_vals_2 = [f2[0, :], f2[1, :]]
    y_vals_1 = [f1[0, :], f1[1, :]]

    for ax, title, y1, y2 in zip(axes.flat, titles, y_vals_1, y_vals_2):
        if (i == 0):
            ax.plot(x1[0, :], y1, 'bo', label='true drift')
        ax.plot(x2[0, :], y2, label='num time series = '+str(meta_error_list[i][0][0]))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-2, 3, 1))
        ax.set_yticks(np.arange(-5.0, 5.0, 1.))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-5, 5])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions')
plt.savefig('./varying_num_timeseries/plots/drift_comparison.eps', format = 'eps', bbox_inches='tight')
