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
hermite_errors = []
ordinary_errors = []

for i in range(parval):
    ts_mapping.append(int(meta_error_list[i][0][0]))
    hermite_errors.append(meta_error_list[i][5])
    ordinary_errors.append(meta_error_list[i][4])

exp = 'varying_num_timeseries'
threshold = meta_error_list[0][3]
ep(exp, hermite_errors, ordinary_errors, parval, ts_mapping, threshold)

###################################################################################################

# 2) Comparison of true drift function vs estimated drift function
def f(theta, x):
    return (theta[0, 0] + theta[1, 0]*x + theta[2, 0]*(x**2) + theta[3, 0]*(x**3))

x1 = np.arange(-2.0, 5.0, 0.5)
x2 = np.arange(-2.0, 5.0, 0.1)

fig = plt.figure()
ax = fig.gca()
plt.plot(x1, f(np.array(meta_error_list[0][2].ordinary), x1), 'bo', label='true drift')
for i in range(parval):
    plt.plot(x2, f(np.array(meta_error_list[i][1].ordinary), x2), label='num timeseries = ' + str(ts_mapping[i]))

plt.axis([-2.0, 5.0, -50.0, 50.0])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Comparison of true drift function vs estimated drift functions')
plt.grid()
plt.savefig('./varying_num_timeseries/plots/drift_comparison.eps', format = 'eps', bbox_inches='tight')
