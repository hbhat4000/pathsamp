import numpy as np
import newbridge as nb
import parameters as prm

"""
Sample equation is the Lorenz oscillator
dx = sigma(y - x) dt + g_1 dWt
dy = (x(rho - z) - y) dt + g_2 dWt
dz = (xy - beta z) dt + g_3 dWt

sigma = 10, rho = 28, beta = 8/3
dimension = 3
degree of freedom for 3rd order hermite polynomial = 64
"""
theta = np.zeros((prm.dof, prm.dim))
gvec = np.array([0.01, 0.01, 0.01])

sigma = 10
rho = 28
beta = 8/3
norm_constant = 0.63161878
norm_constant_sq = np.power(norm_constant, 2)

d_param = prm.data(theta, gvec)
d_param.theta[4, 0] = sigma / norm_constant 	# sigma * y 	in f0
d_param.theta[1, 0] = -sigma / norm_constant	# sigma * x 	in f0
d_param.theta[1, 1] = rho / norm_constant	 	# rho * x 		in f1
d_param.theta[13, 1] = -1. / norm_constant_sq	# x * z 		in f1
d_param.theta[4, 1] = -1. / norm_constant 		# y 			in f1
d_param.theta[19, 2] = 1. / norm_constant_sq 	# x * y 		in f2
d_param.theta[7, 2] = -beta / norm_constant 	# beta * z 		in f2 

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[8.2, 4.1, 1.3], [5.3, 2.2, 0.9], [7.2, -1.8, 2.1], [-0.9, -1.4, 0.4]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = nb.createpaths(d_param, euler_param)

# save to file
import pickle
with open('nem_3D.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, d_param, euler_param], f)