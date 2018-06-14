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

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 11, 1))
    ax.set_yticks(np.arange(-1.0, 3.0, 0.5))

    for j in range(10):
        ic = ((np.round(euler_param.ic[j], 2)))
        plt.plot(t[j, :], x[j, :, 0], label='ic:' + str(ic))

    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.title('Observed data used for random time point experiments in 1D, number of time steps = ' + str(i*10+1))
    plt.grid()
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

################################################################################################

# 3) Comparison of true drift function vs estimated drift function
def f(theta, x):
    return (theta[0, 0] + theta[1, 0]*x + theta[2, 0]*(x**2) + theta[3, 0]*(x**3))

x_true = np.arange(-2.0, 5.0, 0.5)
x_est = np.arange(-2.0, 5.0, 0.1)

fig = plt.figure()
ax = fig.gca()
plt.plot(x_true, f(np.array(meta_error_list[0][2].ordinary), x_true), 'bo', label='true drift')
for i in range(parval):
    plt.plot(x_est, f(np.array(meta_error_list[i][1].ordinary), x_est), label='time points = '+str(meta_error_list[i][0][1]))

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Comparison of true drift function vs estimated drift functions in 1D')
plt.grid()
plt.savefig('./random_timepoints/plots/drift_comparison.pdf', format = 'pdf', bbox_inches='tight')
plt.close()
