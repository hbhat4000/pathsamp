import numpy as np
import newbridge as nb
import parameters as prm

"""
Sample equation is the damped Duffing oscillator
x'' + delta x' + alpha x + beta x^3 = gamma cos(omega t) = gamma (1 - 0.5(omega t)^2)

The first order system of equations are:
dx = y dt + g_1 dWt
dy = (alpha x - beta x^3 - delta y + gamma (1 - 0.5(omega z)^2)) dt + g_2 dWt
dz = 1 dt + g_3 dWt

Standard parameter values:
alpha = -1, beta = +1, gamma = 0.2 to 0.65, delta = 0.3, omega = 1.2
"""
theta = np.zeros((prm.dof, prm.dim))
gvec = np.array([0.1, 0.1, 0.1])

alpha = 1.
beta = 5.
gamma = 8.
delta = 0.02
omega = 0.5
norm_constant = np.power(2. * np.pi, -0.25) 
# norm_constant = 1.

d_param = prm.data(theta, gvec)
d_param.theta[4, 0] = 1. / norm_constant 		# y 			in f0
d_param.theta[1, 1] = alpha / norm_constant 	# alpha * x 	in f1
d_param.theta[3, 1] = -beta / norm_constant 	# beta * x^3 	in f1
d_param.theta[4, 1] = -delta / norm_constant 	# delta * y 	in f1
d_param.theta[0, 1] = gamma / norm_constant 	# gamma 		in f1
d_param.theta[8, 1] = - 0.5 * gamma * np.power(omega, 2) / norm_constant # -gamma * omega^2 / 2 in f1
d_param.theta[0, 2] = 1. / norm_constant 		# 1 			in f2

print(d_param.theta)
# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = np.array([[0.8, 0.5, 0.], [0.4, 0.2, 0.], [-0.25, 0.1, 0.], [1.0, 0.5, 0.]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout, xout_without_noise = nb.createpaths(d_param, euler_param)

# save to file
import pickle
with open('nem_3D_duffing_hermite.pkl','wb') as f:
    pickle.dump([xout, tout, xout_without_noise, d_param, euler_param], f)