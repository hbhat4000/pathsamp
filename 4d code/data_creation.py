import numpy as np
import parameters as prm

def system_drift(sim_param, x):
    derivatives = np.zeros((x.shape[0], x.shape[1]))
    const1 = sim_param.g / sim_param.l
    const2 = sim_param.k / sim_param.m
    derivatives[:, 0] = x[:, 1]
    derivatives[:, 1] = -const1 * np.sin(x[:, 0]) - const2*(x[:, 0] - x[:, 2])
    derivatives[:, 2] = x[:, 3]
    derivatives[:, 3] = -const1 * np.sin(x[:, 2]) - const2*(x[:, 0] - x[:, 2])
    return derivatives 

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
        # k-th initial condition to start off current x and t;''
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
