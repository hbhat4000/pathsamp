import numpy as np
import parameters as prm

"""
Sample equation is the simple oscillator
dx = (alpha + beta x + gamma x^2) dt + g_1 dWt
"""
def system_drift(sim_param, x):
    derivatives = np.zeros((x.shape[0], x.shape[1]))
    derivatives[:, 0] = sim_param.alpha + sim_param.beta * x[:, 0] + sim_param.gamma * np.power(x[:, 0], 2)
    return derivatives 

def true_theta(sim_param):
    theta = np.zeros((prm.dof, prm.dim))
    theta[0, 0] = sim_param.alpha
    theta[1, 0] = sim_param.beta
    theta[2, 0] = sim_param.gamma
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
