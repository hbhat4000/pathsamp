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
gvec = 0.*np.array([0.25, 0.25, 0.25])

sigma = 10
rho = 28
beta = 8/3
nrmlz = 0.63161878
nrmlz2 = nrmlz**2
d_param = prm.data(theta, gvec)
d_param.theta[4, 0] = sigma/nrmlz 	# sigma * y 	in f0
d_param.theta[1, 0] = -sigma/nrmlz 	# sigma * x 	in f0
d_param.theta[1, 1] = rho/nrmlz	 	# rho * x 		in f1
d_param.theta[13, 1] = -1./nrmlz2	# x * z 		in f1
d_param.theta[4, 1] = -1./nrmlz 	# y 			in f1
d_param.theta[10, 2] = 1./nrmlz2 	# x * y 		in f2
d_param.theta[7, 2] = -beta/nrmlz 	# beta * z 		in f2 

# create paths
"""
The default parameters for Euler-Maruyama are:
euler_param = prm.euler_maruyama(numsteps = 25000, savesteps = 100, ft = 10., ic, it, numpaths)
"""
ic = 7.8*np.array([[1., 1., 1.]])
it = np.zeros((ic.shape[0]))
euler_param = prm.euler_maruyama(ic, it)
xout, tout = nb.createpaths(d_param, euler_param)

# save to file
import pickle
with open('nem_3D_nrmlz.pkl','wb') as f:
    pickle.dump([xout, tout, d_param, euler_param], f)
