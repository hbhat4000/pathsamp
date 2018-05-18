import numpy as np
import pickle
import newbridge as nb
from matplotlib import pyplot as plt

# 1) Error plots
meta_error_list = []
for i in range(1, 11):
    with open('./varying_subintervals/tp_51/subint_' + str(i) + '.pkl','rb') as f:
        x, t, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param = pickle.load(f)
    meta_error_list.append((x.shape, error_list, theta_list, estimated_theta, true_theta, inferred_gvec, errors, em_param, data_param, euler_param, sim_param))

parval = 10
error_plot = np.zeros((3, parval))
subint = np.zeros(parval)

for i in range(parval):
    subint[i] = i+1
    error_plot[0, i] = meta_error_list[i][6][1]
    error_plot[1, i] = meta_error_list[i][6][0]
    error_plot[2, i] = np.sqrt(np.sum(np.square(np.abs(meta_error_list[i][6][4])), axis=0))

# 1a) Error in estimated theta in Hermite space
fig = plt.figure()
ax = fig.gca()
plt.plot(subint, error_plot[0, ])
plt.title('Frobenius norm error in estimated theta in Hermite space')
plt.grid()
ax.set_xticks(subint)
# ax.set_ylim([0., 1.])
# ax.set_yticks(np.arange(0., 1.1, 0.1))
plt.savefig('./varying_subintervals/plots/tp_51/hermite.eps', format = 'eps', bbox_inches='tight')

# 1b) Error in estimated theta in Ordinary space
fig = plt.figure()
ax = fig.gca()
plt.plot(subint, error_plot[1, ])
plt.title('Frobenius norm error in estimated theta in Ordinary space')
plt.grid()
ax.set_xticks(subint)
# ax.set_ylim([0., 2.])
# ax.set_yticks(np.arange(0., 2.1, 0.2))
plt.savefig('./varying_subintervals/plots/tp_51/ordinary.eps', format = 'eps', bbox_inches='tight')

# 1c) Error in estimated gvec
fig = plt.figure()
ax = fig.gca()
plt.plot(subint, error_plot[2, ])
plt.title('Frobenius norm error in estimated gvec')
plt.grid()
ax.set_xticks(subint)
# ax.set_ylim([0., 0.05])
# ax.set_yticks(np.arange(0., 0.06, 0.01))
plt.savefig('./varying_subintervals/plots/tp_51/gvec.eps', format = 'eps', bbox_inches='tight')
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

    for d in range(0, 4):
        for k in range(0, d + 1):
            for j in range(0, d + 1):
                for i in range(0, d + 1):
                    if (i + j + k == d):
                        y += theta[index] * np.power(x[0, :], i) * np.power(x[1, :], j) * np.power(x[2, :], k)
                        index += 1

    return y

x_sparse = np.arange(-2.0, 2.0, 0.5)
x_dense = np.arange(-2.0, 2.0, 0.1)
x1 = np.array((x_sparse, x_sparse, x_sparse))
x2 = np.array((x_dense, x_dense, x_dense))

fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True)
fig.set_figwidth(20)
fig.set_figheight(5)
titles = [r'$x_0$', r'$x_1$', r'$x_2$']

for i in range(parval):
    f2 = f(np.array(meta_error_list[i][3].ordinary), x2)
    f1 = f(np.array(meta_error_list[i][4].ordinary), x1)
    y_vals_2 = [f2[0, :], f2[1, :], f2[2, :]]
    y_vals_1 = [f1[0, :], f1[1, :], f1[2, :]]

    for ax, title, y1, y2 in zip(axes.flat, titles, y_vals_1, y_vals_2):
        if (i == 0):
            ax.plot(x1[0, :], y1, 'bo', label='true drift')
        ax.plot(x2[0, :], y2, label='num intervals = '+str(meta_error_list[i][7].numsubintervals))
        ax.set_title(title)
        ax.grid(True)
        ax.set_xticks(np.arange(-2., 3., 1.))
        ax.set_yticks(np.arange(-5.0, 5.0, 1.))
        ax.set_xlim([-2, 2])
        ax.set_ylim([-5, 5])

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.suptitle('Comparison of true drift function vs estimated drift functions')
plt.savefig('./varying_subintervals/plots/tp_51/drift_comparison.eps', format = 'eps', bbox_inches='tight')
