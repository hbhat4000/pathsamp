from matplotlib import pyplot as plt

# load data
import pickle
with open('./varying_subintervals/data/common_data.pkl','rb') as f:
    xout, tout, x_without_noise, euler_param, sim_param = pickle.load(f)

plt.axis([0, 10, -0.5, 3])

for i in range(10):
    plt.plot(tout[0, :], xout[i, :, 0], label='time series '+str(i))

plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.show()
