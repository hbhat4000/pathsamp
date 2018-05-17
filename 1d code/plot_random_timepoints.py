import numpy as np
import pickle
from matplotlib import pyplot as plt
import os

# 1) Data used
for parvalue in range(1, 11):
    with open('./random_timepoints/rand_' + str(parvalue*10+1) + '.pkl', 'rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(-1.0, 3.0, 0.5))

    for i in range(10):
        plt.plot(t[i, :], x[i, :, 0], label='initial condition ' + str(euler_param.ic[i]))

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
error_plot = np.zeros((3, parval))
numpoints = np.zeros(parval)
for i in range(parval):
    numpoints[i] = meta_error_list[i][0][1]
    error_plot[0, i] = meta_error_list[i][6][0]
    error_plot[1, i] = meta_error_list[i][6][1]
    error_plot[2, i] = np.abs(meta_error_list[i][6][4])

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
fig = plt.figure()
ax = fig.gca()
plt.plot(numpoints, error_plot[2, ])
plt.title('Error in estimated gvec')
plt.grid()
ax.set_xticks(numpoints)
plt.savefig('./random_timepoints/plots/error/gvec.eps', format = 'eps', bbox_inches='tight')
###################################################################################################

# 3) Comparison of true drift function vs estimated drift function
def f(theta, x):
    return (theta[0, 0] + theta[1, 0]*x + theta[2, 0]*(x**2) + theta[3, 0]*(x**3))

x1 = np.arange(-2.0, 5.0, 0.5)
x2 = np.arange(-2.0, 5.0, 0.1)

fig = plt.figure()
ax = fig.gca()
plt.plot(x1, f(np.array(meta_error_list[0][4].ordinary), x1), 'bo', label='true drift')
for i in range(parval):
    plt.plot(x2, f(np.array(meta_error_list[i][3].ordinary), x2), label='time points = '+str(meta_error_list[i][0][1]))

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Comparison of true drift function vs estimated drift functions')
plt.grid()
plt.savefig('./random_timepoints/plots/error/drift_comparison.eps', format = 'eps', bbox_inches='tight')
