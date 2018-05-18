import numpy as np
import pickle
from matplotlib import pyplot as plt

# 1) Data used
for parvalue in range(1, 11):
    with open('./random_timepoints/rand_' + str(parvalue*10+1) + '.pkl', 'rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)

    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    titles = [r'$x_0$', r'$x_1$']

    for i in range(10):
        y_vals = [x[i, :, 0], x[i, :, 1]]
        for ax, title, y in zip(axes.flat, titles, y_vals):
            ax.plot(t[i, :], y, label='initial condition '+str(euler_param.ic[i]))
            ax.set_title(title)
            ax.grid(True)
            ax.set_xticks(np.arange(0, 11, 1))
            ax.set_yticks(np.arange(-2.0, 2.0, 0.5))
            ax.set_xlim([0, 10])
            ax.set_ylim([-2, 2])

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.title('Observed data used for random time point experiments, number of time steps = ' + str(parvalue*10+1))
    plt.grid()
    plt.savefig('./random_timepoints/plots/rand_' + str(parvalue*10+1) + '.eps', format = 'eps', bbox_inches='tight')

################################################################################################

# 2) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./random_timepoints/rand_' + str(i*10+1) + '.pkl','rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param))

parval = meta_error_list[0][0][0]
error_plot = np.zeros((3, parval))
numpoints = np.zeros(parval)

for i in range(parval):
    numpoints[i] = meta_error_list[i][0][1]
    error_plot[0, i] = meta_error_list[i][6][1]
    error_plot[1, i] = meta_error_list[i][6][0]
    error_plot[2, i] = np.sqrt(np.sum(np.square(np.abs(meta_error_list[i][6][4])), axis=0))

# 2a) Error in estimated theta in Hermite space
fig = plt.figure()
ax = fig.gca()
plt.plot(numpoints, error_plot[0, ])
plt.title('Frobenius norm error in estimated theta in Hermite space')
plt.grid()
ax.set_xticks(numpoints)
# ax.set_ylim([0., 1.])
# ax.set_yticks(np.arange(0., 1.1, 0.1))
plt.savefig('./random_timepoints/plots/hermite.eps', format = 'eps', bbox_inches='tight')

# 2b) Error in estimated theta in Ordinary space
fig = plt.figure()
ax = fig.gca()
plt.plot(numpoints, error_plot[1, ])
plt.title('Frobenius norm error in estimated theta in Ordinary space')
plt.grid()
# ax.set_ylim([0., 2.])
# ax.set_yticks(np.arange(0., 2.1, 0.2))
ax.set_xticks(numpoints)
plt.savefig('./random_timepoints/plots/ordinary.eps', format = 'eps', bbox_inches='tight')

# 2c) Error in estimated gvec
fig = plt.figure()
ax = fig.gca()
plt.plot(numpoints, error_plot[2, ])
plt.title('Frobenius norm error in estimated gvec')
plt.grid()
ax.set_xticks(numpoints)
# ax.set_ylim([0., 0.05])
# ax.set_yticks(np.arange(0., 0.06, 0.01))
plt.savefig('./random_timepoints/plots/gvec.eps', format = 'eps', bbox_inches='tight')
###################################################################################################

# 3) Comparison of true drift function vs estimated drift function
def f(theta, x):
    y = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        y[i, :] = theta[0, i] + theta[1, i]*x[0, :] + theta[2, i]*x[1, :] + theta[3, i]*(x[0, :]**2) + theta[4, i]*(x[0, :]*x[1, :]) + theta[5, i]*(x[1, :]**2) + theta[6, i]*(x[0, :]**3) + theta[7, i]*(x[0, :]**2 * x[1, :]) + theta[8, i]*(x[0, :] * x[1, :]**2) + theta[9, i]*(x[1, :]**3)
    return (y)

x1 = np.array((np.arange(-2.0, 2.0, 0.5), np.arange(-2.0, 2.0, 0.5)))
x2 = np.array((np.arange(-2.0, 2.0, 0.1), np.arange(-2.0, 2.0, 0.1)))

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(5)
titles = [r'$x_0$', r'$x_1$']

for i in range(parval):
    y_vals_2 = [f(np.array(meta_error_list[i][3].ordinary), x2)[0, :], f(np.array(meta_error_list[i][3].ordinary), x2)[1, :]]
    y_vals_1 = [f(np.array(meta_error_list[0][4].ordinary), x1)[0, :], f(np.array(meta_error_list[0][4].ordinary), x1)[1, :]]

    for ax, title, y1, y2 in zip(axes.flat, titles, y_vals_1, y_vals_2):
        if (i == 0):
            ax.plot(x1[0, :], y1, 'bo', label='true drift')
        ax.plot(x2[0, :], y2, label='time points = '+str(meta_error_list[i][0][1]))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-2, 3, 1))
        ax.set_yticks(np.arange(-5.0, 5.0, 1.))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-5, 5])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions')
plt.savefig('./random_timepoints/plots/drift_comparison.eps', format = 'eps', bbox_inches='tight')