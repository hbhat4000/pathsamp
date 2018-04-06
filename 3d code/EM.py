import numpy as np
import newbridge as nb
import parameters as prm

# load data
import pickle
with open('nem_3D_duffing_trueDrift.pkl','rb') as f:
    allx, allt, x_without_noise, euler_param, sim_param = pickle.load(f)

"""
dimension = 3
degree of freedom for 3rd order hermite polynomial = 64
"""

theta = np.zeros((prm.dof, prm.dim))
gvec = np.array([0.01, 0.01, 1e-7])
d_param = prm.data(theta, gvec)
norm_constant = np.power(2. * np.pi, -0.25) 

# normalized coeffecients for the Hermite function
d_param.theta[2, 0] = 1. / norm_constant 				# y 			in f0
d_param.theta[3, 1] = -(sim_param.alpha + 3*sim_param.beta) / norm_constant 	# alpha * x => -(alpha + 3*beta) / h1	in f1
d_param.theta[19, 1] = -sim_param.beta / (norm_constant / np.sqrt(6)) 	# beta * x^3 => -beta / h3	in f1
d_param.theta[2, 1] = -sim_param.delta / norm_constant 	# delta * y 	in f1
d_param.theta[0, 1] = sim_param.gamma / norm_constant 	# gamma 		in f1
d_param.theta[8, 1] = - 0.5 * sim_param.gamma * np.power(sim_param.omega, 2) / norm_constant # -gamma * omega^2 / 2 in f1
d_param.theta[0, 2] = 1. / norm_constant 				# 1 			in f2

print("Data shape:", allx.shape)
print("Theta shape:", d_param.theta.shape)

# theta = 0.1 * np.random.randn(prm.dof, prm.dim)
data_param = prm.data(theta = d_param.theta, gvec = d_param.gvec)

"""
Default parameters for Expectation-Maximization
em_param = param.em(tol = 1e-3, burninpaths = 10, mcmcpaths = 100, numsubintervals = 10, niter = 100, dt = (allt[0, 1, 0] - allt[0, 0, 0]))
"""
em_param = prm.em(dt = allt[0, 1] - allt[0, 0])

# call to EM which returns the final error and estimated theta value
error, d_param = nb.em(allx, allt, em_param, data_param)

print("Error", error)
print("Theta: ", d_param.theta)
print("gvec:", d_param.gvec)