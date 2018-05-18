import numpy as np
import pickle
from matplotlib import pyplot as plt
import newbridge as nb

# 1) Data used
for parvalue in range(1, 11):
    with open('./random_timepoints/rand_' + str(parvalue*10+1) + '.pkl', 'rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)

    fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
    fig.set_figwidth(20)
    fig.set_figheight(15)
    titles = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$']

    for i in range(10):
        y_vals = [x[i, :, 0], x[i, :, 1], x[i, :, 2], x[i, :, 3], x[i, :, 4], x[i, :, 5]]
        for ax, title, y in zip(axes.flat, titles, y_vals):
            ax.plot(t[i, :], y, label='initial condition '+str(euler_param.ic[i]))
            ax.set_title(title)
            ax.grid(True)
            # ax.set_xticks(np.arange(0, 11, 1))
            # ax.set_yticks(np.arange(-2.0, 2.0, 0.5))
            # ax.set_xlim([0, 10])
            # ax.set_ylim([-2, 2])

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.title('Observed data used for random time point experiments, number of time steps = ' + str(parvalue*10+1))
    plt.grid()
    plt.savefig('./random_timepoints/plots/data/rand_' + str(parvalue*10+1) + '.eps', format = 'eps', bbox_inches='tight')

################################################################################################

# 2) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./random_timepoints/rand_' + str(i*10+1) + '.pkl','rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param))

parval = meta_error_list[0][0][0]
error_plot = np.zeros((2, parval))
g_error = np.zeros((6, parval))
numpoints = np.zeros(parval)

for i in range(parval):
    numpoints[i] = meta_error_list[i][0][1]
    error_plot[0, i] = meta_error_list[i][6][0]
    error_plot[1, i] = meta_error_list[i][6][1]
    g_error[:, i] = np.abs(meta_error_list[i][6][4])

print(error_plot)
# 2a) Error in estimated theta in Hermite space
fig = plt.figure()
ax = fig.gca()
plt.plot(numpoints, error_plot[0, ])
plt.title('Error in estimated theta in Hermite space')
plt.grid()
ax.set_xticks(numpoints)
plt.savefig('./random_timepoints/plots/error/hermite.eps', format = 'eps', bbox_inches='tight')

# 2b) Error in estimated theta in Ordinary space
fig = plt.figure()
ax = fig.gca()
plt.plot(numpoints, error_plot[1, ])
plt.title('Error in estimated theta in Ordinary space')
plt.grid()
ax.set_xticks(numpoints)
plt.savefig('./random_timepoints/plots/error/ordinary.eps', format = 'eps', bbox_inches='tight')

# 2c) Error in estimated gvec
fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(15)
titles = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$']

y_vals = [g_error[0, :], g_error[1, :], g_error[2, :], g_error[3, :], g_error[4, :], g_error[5, :]]
for ax, title, y in zip(axes.flat, titles, y_vals):
    ax.plot(numpoints, y)
    ax.set_title(title)
    ax.grid(True)
    ax.set_xticks(numpoints)

plt.suptitle('Error in estimated gvec')
plt.savefig('./random_timepoints/plots/error/gvec.eps', format = 'eps', bbox_inches='tight')

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

    for d in range(0, 4):
        for n in range(0, d + 1):
            for m in range(0, d + 1):
                for l in range(0, d + 1):
                    for k in range(0, d + 1):
                        for j in range(0, d + 1):
                            for i in range(0, d + 1):
                                if (i + j + k + l + m + n == d):
                                    y += theta[index] * np.power(x[0, :], i) * np.power(x[1, :], j) * np.power(x[2, :], k) * np.power(x[3, :], l) * np.power(x[4, :], m) * np.power(x[5, :], n)
                                    index += 1

    return y

x_sparse = np.arange(-2.0, 2.0, 0.5)
x_dense = np.arange(-2.0, 2.0, 0.1)
x1 = np.array((x_sparse, x_sparse, x_sparse, x_sparse, x_sparse, x_sparse))
x2 = np.array((x_dense, x_dense, x_dense, x_dense, x_dense, x_dense))

fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True)
fig.set_figwidth(20)
fig.set_figheight(15)
titles = [r'$x_0$', r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$']

for i in range(parval):
    f1 = f(np.array(meta_error_list[i][3].ordinary), x1)
    f2 = f(np.array(meta_error_list[0][4].ordinary), x2)

    y_vals_1 = [f1[0, :], f1[1, :], f1[2, :], f1[3, :], f1[4, :], f1[5, :]]
    y_vals_2 = [f2[0, :], f2[1, :], f2[2, :], f2[3, :], f2[4, :], f2[5, :]]

    for ax, title, y1, y2 in zip(axes.flat, titles, y_vals_1, y_vals_2):
        if (i == 0):
            ax.plot(x1[0, :], y1, 'bo', label='true drift')
        ax.plot(x2[0, :], y2, label='time points = '+str(meta_error_list[i][0][1]))
        ax.set_title(title)
        ax.grid(True)
        # ax.set_xticks(np.arange(-5, 5, 1))
        # ax.set_yticks(np.arange(-50.0, 50.0, 10.))
        # ax.set_xlim([-5, 5])
        # ax.set_ylim([-50, 50])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions')
plt.savefig('./random_timepoints/plots/error/drift_comparison.eps', format = 'eps', bbox_inches='tight')
