from matplotlib import pyplot as plt

# load data
import pickle
with open('1D_results_noise1.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise2.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise3.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise4.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise5.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise6.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise7.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise8.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)
with open('1D_results_noise9.pkl','rb') as f:
    error_list, theta_list, em_param, data_param = pickle.load(f)

p0, = plt.plot(tout[0, :], xout[0, :, 0], label='time series 0')
p1, = plt.plot(tout[0, :], xout[0, :, 0], label='time series 1')
p2, = plt.plot(tout[0, :], xout[0, :, 0], label='time series 2')
p3, = plt.plot(tout[0, :], xout[0, :, 0], label='time series 3')
p4, = plt.plot(tout[0, :], xout[0, :, 0], label='time series 4')
p5, = plt.plot(tout[0, :], xout[5, :, 0], label='time series 5')
p6, = plt.plot(tout[0, :], xout[6, :, 0], label='time series 6')
p7, = plt.plot(tout[0, :], xout[7, :, 0], label='time series 7')
p8, = plt.plot(tout[0, :], xout[8, :, 0], label='time series 8')
p9, = plt.plot(tout[0, :], xout[9, :, 0], label='time series 9')
p10, = plt.plot(tout[0, :], xout[10, :, 0], label='time series 10')
p11, = plt.plot(tout[0, :], xout[11, :, 0], label='time series 11')
p12, = plt.plot(tout[0, :], xout[12, :, 0], label='time series 12')
plt.legend(handles = [p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12], bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Damped Duffing oscillator (x\')')
plt.show()
