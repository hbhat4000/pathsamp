import numpy as np
import pickle
from error_plots import error_plots as ep
from matplotlib import pyplot as plt

# 1) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./varying_num_timeseries/ts_' + str(i) + '.pkl','rb') as f:
        x, t, error_list, theta_list, gammavec_list, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, estimated_theta, true_theta, threshold, ordinary_errors, hermite_errors))

parval = 10
ts_mapping = []

for i in range(parval):
    ts_mapping.append(int(meta_error_list[i][0][0]))

hermite_errors = np.zeros((threshold.shape[0], 6, parval))
ordinary_errors = np.zeros((threshold.shape[0], 6, parval))

for th in range(threshold.shape[0]):
	for fn in range(6):
		for val in range(parval):
			hermite_errors[th][fn][val] = meta_error_list[val][5][th][fn]
			ordinary_errors[th][fn][val] = meta_error_list[val][4][th][fn]

exp = 'varying_num_timeseries'
threshold = meta_error_list[0][3]
ep(exp, hermite_errors, ordinary_errors, parval, ts_mapping, threshold)

###################################################################################################

# 2) Comparison of true drift function vs estimated drift function
def f(theta, x):
    return (theta[0, 0] + theta[1, 0]*x + theta[2, 0]*(x**2) + theta[3, 0]*(x**3))

x_true = np.arange(-2.0, 5.0, 0.5)
x_est = np.arange(-2.0, 5.0, 0.1)

fig = plt.figure()
ax = fig.gca()
plt.plot(x_true, f(np.array(meta_error_list[0][2].ordinary), x_true), 'bo', label='true drift')
for i in range(parval):
    plt.plot(x_est, f(np.array(meta_error_list[i][1].ordinary), x_est), label='num timeseries = '+str(ts_mapping[i]))

plt.axis([-2.0, 5.0, -50.0, 50.0])
# plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Comparison of true drift function vs estimated drift functions in 1D')
plt.grid()
plt.savefig('./varying_num_timeseries/plots/drift_comparison.pdf', format = 'pdf', bbox_inches='tight')
plt.close()
