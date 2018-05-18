import numpy as np
import pickle
from matplotlib import pyplot as plt

# 1) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./varying_num_timeseries/ts_' + str(i) + '.pkl','rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param))

parval = 10
error_plot = np.zeros((3, parval))
ts = np.zeros(parval)
for i in range(parval):
    ts[i] = meta_error_list[i][0][0]
    error_plot[0, i] = meta_error_list[i][6][1]
    error_plot[1, i] = meta_error_list[i][6][0]
    error_plot[2, i] = np.sqrt(np.sum(np.square(np.abs(meta_error_list[i][6][4]))))

# 1a) Error in estimated theta in Hermite space
fig = plt.figure()
ax = fig.gca()
plt.plot(ts, error_plot[0, ])
plt.title('Frobenius norm error in estimated theta in Hermite space')
plt.grid()
ax.set_xticks(ts)
# ax.set_ylim([0., 1.])
# ax.set_yticks(np.arange(0., 1.1, 0.1))
plt.savefig('./varying_num_timeseries/plots/hermite.eps', format = 'eps', bbox_inches='tight')

# 1b) Error in estimated theta in Ordinary space
fig = plt.figure()
ax = fig.gca()
plt.plot(ts, error_plot[1, ])
plt.title('Frobenius norm error in estimated theta in Ordinary space')
plt.grid()
ax.set_xticks(ts)
# ax.set_ylim([0., 2.])
# ax.set_yticks(np.arange(0., 2.1, 0.2))
plt.savefig('./varying_num_timeseries/plots/ordinary.eps', format = 'eps', bbox_inches='tight')

# 1c) Error in estimated gvec
fig = plt.figure()
ax = fig.gca()
plt.plot(ts, np.abs(error_plot[2, ]))
plt.title('Frobenius norm error in estimated gvec')
plt.grid()
ax.set_xticks(ts)
# ax.set_ylim([0., 0.05])
# ax.set_yticks(np.arange(0., 0.06, 0.01))
plt.savefig('./varying_num_timeseries/plots/gvec.eps', format = 'eps', bbox_inches='tight')
###################################################################################################

# 2) Comparison of true drift function vs estimated drift function
def f(theta, x):
    return (theta[0, 0] + theta[1, 0]*x + theta[2, 0]*(x**2) + theta[3, 0]*(x**3))

x1 = np.arange(-2.0, 5.0, 0.5)
x2 = np.arange(-2.0, 5.0, 0.1)

fig = plt.figure()
ax = fig.gca()
plt.plot(x1, f(np.array(meta_error_list[0][4].ordinary), x1), 'bo', label='true drift')
for i in range(parval):
    plt.plot(x2, f(np.array(meta_error_list[i][3].ordinary), x2), label='num timeseries = '+str(ts[i]))

plt.axis([-2.0, 5.0, -50.0, 50.0])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Comparison of true drift function vs estimated drift functions')
plt.grid()
plt.savefig('./varying_num_timeseries/plots/drift_comparison.eps', format = 'eps', bbox_inches='tight')
