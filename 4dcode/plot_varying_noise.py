import numpy as np
import pickle
from error_plots import error_plots as ep
from tables import error_tables as et
from matplotlib import pyplot as plt
import parameters as prm

# 1) Error plots
meta_error_list = []
for i in range(8):
    with open('./varying_noise/noise_' + str(i) + '.pkl','rb') as f:
        x, t, error_list, theta_list, gammavec_list, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors, sim_param))

parval = 8
noise_mapping = (0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001)
exp = 'varying_noise'
threshold = meta_error_list[0][3]
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

ep(exp, hermite_errors, ordinary_errors, parval, noise_mapping, threshold)
et(exp, hermite_errors, ordinary_errors, estimated_theta, true_theta, parval, noise_mapping, threshold)

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
        for l in range(0, d + 1):
            for k in range(0, d + 1):
                for j in range(0, d + 1):
                    for i in range(0, d + 1):
                        if (i + j + k + l == d):
                            y += theta[index] * np.power(x[0, :], i) * np.power(x[1, :], j) * np.power(x[2, :], k) * np.power(x[3, :], l)
                            index += 1

    return y

x_sparse = np.arange(-2.0, 2.0, 0.5)
x_dense = np.arange(-2.0, 2.0, 0.1)
x_true = np.array((x_sparse, x_sparse, x_sparse, x_sparse))
x_est = np.array((x_dense, x_dense, x_dense, x_dense))

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_figwidth(20)
fig.set_figheight(10)
titles = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$']

for i in range(parval):
    f_true = f(np.array(meta_error_list[i][2].ordinary), x_true)
    f_est = f(np.array(meta_error_list[i][1].ordinary), x_est)
    y_vals_true = [f_true[0, :], f_true[1, :], f_true[2, :], f_true[3, :]]
    y_vals_est = [f_est[0, :], f_est[1, :], f_est[2, :], f_est[3, :]]

    for ax, title, y_true, y_est in zip(axes.flat, titles, y_vals_true, y_vals_est):
        if (i == 0):
            ax.plot(x_true[0, :], y_true, 'bo', label='true drift')
        ax.plot(x_est[0, :], y_est, label='noise = '+str(meta_error_list[i][6].gvec[0]))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-2., 3., 1.))
        ax.set_yticks(np.arange(-5.0, 5.0, 1.))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-5, 5])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions in 4D')
plt.savefig('./varying_noise/plots/drift_comparison.pdf', format = 'pdf', bbox_inches='tight')
plt.close()

#####################################################
# 3) 3D drift comparison

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection = '3d')

x_sparse = np.arange(-2.0, 2.0, 0.5)
x_dense = np.arange(-2.0, 2.0, 0.1)
x_true = np.array((x_sparse, x_sparse, x_sparse, x_sparse))
x_est = np.array((x_dense, x_dense, x_dense, x_dense))

f_true = f(np.array(meta_error_list[0][2].ordinary), x_true)
ax.plot(f_true[1, :], f_true[2, :], f_true[3, :], 'bo--', label='True drift')

for i in range(parval):
    f_est = f(np.array(meta_error_list[i][1].ordinary), x_est)
    ax.plot(f_est[1, :], f_est[2, :], f_est[3, :], label='noise = '+str(meta_error_list[i][6].gvec[0]))

ax.set_xlim([-10., 10.])
ax.set_ylim([-0.04, 0.04])
ax.set_zlim([-10., 10.])
plt.xlabel('x')
plt.ylabel('y')
# plt.zlabel('z')
plt.legend(bbox_to_anchor = (1.05, 1), loc= 2, borderaxespad = 0.)
plt.suptitle('3D comparison of true drift function vs estimated drift functions in 4D system')
plt.savefig('./varying_noise/plots/3D_drift.pdf', format = 'pdf', bbox_inches='tight')
plt.close()
