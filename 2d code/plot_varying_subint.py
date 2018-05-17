import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

# 1) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./varying_subintervals/tp_11/subint_' + str(i) + '.pkl','rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param))

parval = 10
error_plot = np.zeros((2, parval))
g_error = np.zeros((2, parval))
subint = np.zeros(parval)

for i in range(parval):
    subint[i] = meta_error_list[i][7].numsubintervals
    error_plot[0, i] = meta_error_list[i][6][0]
    error_plot[1, i] = meta_error_list[i][6][1]
    g_error[:, i] = np.abs(meta_error_list[i][6][4])

# 1a) Error in estimated theta in Hermite space
fig = plt.figure()
ax = fig.gca()
plt.plot(subint, error_plot[0, ])
plt.title('Error in estimated theta in Hermite space')
plt.grid()
ax.set_xticks(subint)
plt.savefig('./varying_subintervals/plots/tp_11/hermite.eps', format = 'eps', bbox_inches='tight')

# 1b) Error in estimated theta in Ordinary space
fig = plt.figure()
ax = fig.gca()
plt.plot(subint, error_plot[1, ])
plt.title('Error in estimated theta in Ordinary space')
plt.grid()
ax.set_xticks(subint)
plt.savefig('./varying_subintervals/plots/tp_11/ordinary.eps', format = 'eps', bbox_inches='tight')

# 1c) Error in estimated gvec
fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(5)
titles = [r'$x_0$', r'$\dot{x_0}$']

y_vals = [g_error[0, :], g_error[1, :]]
for ax, title, y in zip(axes.flat, titles, y_vals):
    ax.plot(subint, y, label='time points = '+str(meta_error_list[i][0][1]))
    ax.set_title(title)
    ax.grid(True)
    ax.set_xticks(subint)

plt.suptitle('Error in estimated gvec')
plt.savefig('./varying_subintervals/plots/tp_11/gvec.eps', format = 'eps', bbox_inches='tight')

###################################################################################################

# 2) Comparison of true drift function vs estimated drift function
def f(theta, x):
    y = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        y[i, :] = theta[0, i] + theta[1, i]*x[0, :] + theta[2, i]*x[1, :] + theta[3, i]*(x[0, :]**2) + theta[4, i]*(x[0, :]*x[1, :]) + theta[5, i]*(x[1, :]**2) + theta[6, i]*(x[0, :]**3) + theta[7, i]*(x[0, :]**2 * x[1, :]) + theta[8, i]*(x[0, :] * x[1, :]**2) + theta[9, i]*(x[1, :]**3)
    return (y)

x1 = np.array((np.arange(-5.0, 6.0, 1), np.arange(-5.0, 6.0, 1)))
x2 = np.array((np.arange(-5.0, 6.0, 1), np.arange(-5.0, 6.0, 1)))

fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True)
fig.set_figwidth(15)
fig.set_figheight(5)
titles = [r'$x_0$', r'$\dot{x_0}$']

for i in range(parval):
    y_vals_2 = [f(np.array(meta_error_list[i][3].ordinary), x2)[0, :], f(np.array(meta_error_list[i][3].ordinary), x2)[1, :]]
    y_vals_1 = [f(np.array(meta_error_list[0][4].ordinary), x1)[0, :], f(np.array(meta_error_list[0][4].ordinary), x1)[1, :]]
    for ax, title, y1, y2 in zip(axes.flat, titles, y_vals_1, y_vals_2):
        if (i == 0): 
            ax.plot(x1[0, :], y1, 'bo', label='true drift')
        ax.plot(x2[0, :], y2, label='num subintervals = '+str(meta_error_list[i][7].numsubintervals))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-5., 5., 1.))
        ax.set_yticks(np.arange(-50.0, 50.0, 10.))
        ax.set_xlim([-5, 5])
        ax.set_ylim([-50, 50])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions')
plt.savefig('./varying_subintervals/plots/tp_11/drift_comparison.eps', format = 'eps', bbox_inches='tight')
