import numpy as np
import parameters as prm

"""
System of equations:
x_0' = x_1 dt + g_0 dW_0
x_1' = [-k_0/m_0 (x_0) - k_1/m_0 (x_0 - x_2)]dt + g_1 dW_1
x_2' = x_3 dt + g_2 dW_2
x_3' = [-k_1/m_1 (x_2 - x_0) - k_2/m_1 (x_2 - x_4)]dt + g_3 dW_3
x_4' = x_5 dt + g_4 dW_4
x_5' = [-k_2/m_2 (x_4 - x_2) - k_3/m_2 (x_4 - x_6)]dt + g_5 dW_5
x_6' = x_7 dt + g_6 dW_6
x_7' = [-k_3/m_3 (x_6 - x_4) - k_4/m_3 (x_6)]dt + g_7 dW_7

Standard parameters:
kvec = [1., 0.7, 0.6, 1.2, 0.9]
mvec = [0.2, 0.3, 0.5, 1.1]
"""
def system_drift(sim_param, x):
    derivatives = np.zeros((x.shape[0], x.shape[1]))
    derivatives[:, 0] = x[:, 1]
    derivatives[:, 1] = -(sim_param.kvec[0] / sim_param.mvec[0]) * (x[:, 0]) -(sim_param.kvec[1] / sim_param.mvec[0]) * (x[:, 0] - x[:, 2])
    derivatives[:, 2] = x[:, 3]
    derivatives[:, 3] = -(sim_param.kvec[1] / sim_param.mvec[1]) * (x[:, 2] - x[:, 0]) - (sim_param.kvec[2] / sim_param.mvec[1]) * (x[:, 2] - x[:, 4])
    derivatives[:, 4] = x[:, 5]
    derivatives[:, 5] = -(sim_param.kvec[2] / sim_param.mvec[2]) * (x[:, 4] - x[:, 2]) - (sim_param.kvec[3] / sim_param.mvec[2]) * (x[:, 4] - x[:, 6])
    derivatives[:, 6] = x[:, 7]
    derivatives[:, 7] = -(sim_param.kvec[3] / sim_param.mvec[3]) * (x[:, 6] - x[:, 4]) - (sim_param.kvec[4] / sim_param.mvec[3]) * (x[:, 6])

    return derivatives 

def true_theta(sim_param):
    theta = np.zeros((prm.dof, prm.dim))
    theta[2, 0] = 1.
    theta[1, 1] = -(sim_param.kvec[0] + sim_param.kvec[1]) / sim_param.mvec[0]
    theta[3, 1] = sim_param.kvec[1] / sim_param.mvec[0]
    theta[4, 2] = 1.
    theta[1, 3] = sim_param.kvec[1] / sim_param.mvec[1]
    theta[3, 3] = -(sim_param.kvec[1] + sim_param.kvec[2]) / sim_param.mvec[1]
    theta[5, 3] = sim_param.kvec[2] / sim_param.mvec[1]
    theta[6, 4] = 1.
    theta[3, 5] = sim_param.kvec[2] / sim_param.mvec[2]
    theta[5, 5] = -(sim_param.kvec[2] + sim_param.kvec[3]) / sim_param.mvec[2]
    theta[7, 5] = sim_param.kvec[3] / sim_param.mvec[2]
    theta[8, 6] = 1.
    theta[5, 7] = sim_param.kvec[3] / sim_param.mvec[3]
    theta[7, 7] = -(sim_param.kvec[3] + sim_param.kvec[4]) / sim_param.mvec[3]

    return theta

def system_diffusion(sim_param):
    return np.dot(sim_param.gvec, np.random.standard_normal(prm.dim))

# create sample paths! 
# this function creates a bunch of Euler-Maruyama paths from an array
# of initial conditions
def createpaths(euler_param, sim_param):
    h12 = np.sqrt(euler_param.h)

    x = np.zeros(( euler_param.numpaths, (euler_param.savesteps + 1), prm.dim))
    x_without_noise = np.zeros(( euler_param.numpaths, (euler_param.savesteps + 1), prm.dim ))
    t = np.zeros(( euler_param.numpaths, (euler_param.savesteps + 1) ))

    x[:, 0, :] = euler_param.ic
    x_without_noise[:, 0, :] = euler_param.ic
    t[:, 0] = euler_param.it

    # for each time series, generate the matrix of size savesteps * dim
    # corresponding to one 2D time series
    for k in range(euler_param.numpaths):
        # k-th initial condition to start off current x and t
        curx = euler_param.ic[[k]]
        curx_without_noise = euler_param.ic[[k]]
        curt = euler_param.it[k]
        j = 1
        for i in range(1, euler_param.numsteps + 1):
            curx += system_drift(sim_param, curx) * euler_param.h + system_diffusion(sim_param) * h12
            curx_without_noise += system_drift(sim_param, curx_without_noise) * euler_param.h
            curt += euler_param.h
            if (i % (euler_param.numsteps // euler_param.savesteps) == 0):
                x[k, j, :] = curx
                x_without_noise[k, j, :] = curx_without_noise
                t[k, j] = curt
                j += 1

    return x, t, x_without_noise
