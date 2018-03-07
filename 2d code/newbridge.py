import numpy as np
import parameters as prm
from joblib import Parallel, delayed

def H0():
    return 0.63161878

def H1(x):
    return 0.63161878 * x

def H2(x):
    return 0.44662192 * (np.power(x, 2) - 1)

def H3(x):
    return -0.77357185 * x + 0.25785728 * np.power(x, 3)

# this function defines our hermite basis functions
# x must be a numpy array, a column vector of points
# (x = vector of points at which we seek to evaluate the basis functions)
# dof is the number of degrees of freedom, i.e., the number of basis functions
def hermite_basis(x):
    y = np.zeros((prm.dim, prm.dof))

    y[:, 0] = H0()

    index = 1
    for i in range(prm.dim):
        y[:, index] = H1(x[i])
        y[:, index + 1] = H2(x[i])
        y[:, index + 2] = H3(x[i])
        index = 4
    
    index = 7
    for i in range(prm.polynomial_degree):
        for j in range(prm.polynomial_degree - 1):
            y[:, index] = np.dot(y[:, i], y[:, (4 + j)])
            index += 1

    return y

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    y = np.zeros((prm.dim, prm.dof))

    y[:, 0] = 1

    index = 1
    for i in range(prm.dim):
        y[:, index] = x[i]
        y[:, index + 1] = np.power(x[i], 2)
        y[:, index + 2] = np.power(x[i], 3)
        index = 4
    
    index = 7
    for i in range(prm.polynomial_degree):
        for j in range(prm.polynomial_degree - 1):
            y[:, index] = np.dot(y[:, i], y[:, (4 + j)])
            index += 1

    return y

# drift function using "basis" functions defined by mypoly
# x must be a numpy array, the points at which the drift is to be evaluated
# theta must also be a numpy array, the coefficients of each "basis" function
# dof is implicitly calculated based on the first dimension of theta
def drift(x, d_param):
    evaluated_basis = hermite_basis(x)
    out = np.zeros(prm.dim)
    out[0] = np.sum(np.dot(evaluated_basis[0, :], d_param.theta[:, 0]))
    out[1] = np.sum(np.dot(evaluated_basis[1, :], d_param.theta[:, 1]))
    return out

def diffusion(d_param):
    return np.dot(d_param.gvec, np.random.standard_normal(prm.dim))

# create sample paths! 

# this function creates a bunch of Euler-Maruyama paths from an array
# of initial conditions

# coef = coefficients of drift function in the hermite basis
# g = diffusion coefficient
# numsteps = total number of internal time steps to take
# h = size of each internal time step
# savesteps = number of times to save the solution
# ic = vector of initial conditions
# it = corresponding vector of initial times
def createpaths(d_param, euler_param):
    h12 = np.sqrt(euler_param.h)
    numpaths = euler_param.ic.shape[0]

    x = np.zeros(( numpaths, (euler_param.savesteps + 1), prm.dim))
    t = np.zeros(( numpaths, (euler_param.savesteps + 1), prm.dim))

    x[:, 0, :] = euler_param.ic
    t[:, 0, :] = euler_param.it

    # for each time series, generate the matrix of size savesteps * dim
    # corresponding to one 2D time series
    for k in range(numpaths):
        # k-th initial condition to start off current x and t
        curx = euler_param.ic[k, :]
        curt = euler_param.it[k, :]
        j = 1
        for i in range(1, euler_param.numsteps + 1):
            curx += drift(curx, d_param) * euler_param.h + diffusion(d_param) * h12
            curt += euler_param.h
            if (i % (euler_param.numsteps // euler_param.savesteps) == 0):
                x[k, j, :] = curx
                t[k, j, :] = curt
                j += 1

    return x, t

# creates brownian bridge interpolations for given start and end
# time point t and value x.
def brownianbridge(gvec, xin, tin, n):
    h = (tin[1] - tin[0]) / n
    tvec = tin[0] + (1 + np.arange(n))*h
    h12 = np.sqrt(h)

    # W ~ N(0, sqrt(h)*g)
    wincs = np.random.multivariate_normal(mean = np.zeros(prm.dim), 
        cov = h * np.diag(np.square(gvec)), 
        size = n)
    w = np.cumsum(wincs, axis = 0).T

    bridge = xin[0, :, None] + w
    bridge -= ((tvec - tin[0])/(tin[1]-tin[0]))*(w[:,n-1,None] + xin[0,:,None] - xin[1,:,None])
    
    # concatenate the starting point to the bridge
    tvec = np.concatenate((tin[[index]], tvec))
    bridge = np.concatenate((xin[[index]], bridge))
    return tvec, bridge

# Girsanov likelihood is computed using # TODO: insert reference to the paper
def girsanov(g, path, dt, theta):
    b = drift(path, theta)
    int1 = np.dot(b[:-1]/(g*g), np.diff(path))
    b2 = np.square(b)/(g*g)
    int2 = np.sum(0.5*(b2[1:] + b2[:-1]))*dt
    r = int1 - 0.5*int2
    return r

# this function computes MCMC steps for i-th interval of the j-th time series
# using Brownian bridge. The accept-reject step is computed using the Girsanov
# likelihood function. First burnin steps are rejected and the next numsteps
# are used to compute the mmat and rvec (E step) which are used to solve the system of 
# equations producing the next iteration of theta (M step).
def mcmc2D(allx, allt, em_param, d_param, path_index, step_index, dim_index):
    mmat = np.zeros((prm.dim, prm.dof, prm.dof))
    rvec = np.zeros((prm.dim, prm.dof))

    # one time series, one interval, one dimension at a time
    x = allx[path_index, step_index:(step_index + 2), dim_index]
    t = allt[path_index, step_index:(step_index + 2), dim_index]
    
    samples = np.zeros(em_param.numsubintervals)
    _, xcur = brownianbridge(d_param.gvec, x, t, em_param.numsubintervals)
    oldlik = girsanov(d_param.gvec, path = xcur, dt = em_param.h, theta = d_param.theta)
    arburn = np.zeros(em_param.burninpaths)
    for jj in range(em_param.burninpaths):
        _, prop = brownianbridge(d_param.gvec, x, t, em_param.numsubintervals)
        proplik = girsanov(d_param.gvec, path = prop, dt = em_param.h, theta = d_param.theta)
        rho = np.exp(proplik - oldlik)
        if (rho > np.random.uniform()):
            xcur = prop
            oldlik = proplik
            arburn[jj] = 1
    meanBurnin = np.mean(arburn)
    
    # for each path being sampled (r = 0 to r = R)
    arsamp = np.zeros(em_param.mcmcpaths)
    for jj in range(em_param.mcmcpaths):
        _, prop = brownianbridge(d_param.gvec, x, t, em_param.numsubintervals)
        proplik = girsanov(d_param.gvec, path = prop, dt = em_param.h, theta = d_param.theta)
        rho = np.exp(proplik - oldlik)
        if (rho > np.random.uniform()):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1
        samples = xcur
        pp = hermite_basis(samples[:(-1)])
        mmat = mmat + em_param.h * np.matmul(pp.T, pp) / em_param.mcmcpaths
        rvec = rvec + np.matmul((np.diff(samples)).T, pp) / em_param.mcmcpaths   
    meanSample = np.mean(arsamp)
    
    return (mmat, rvec, meanBurnin, meanSample, path_index, step_index, dim_index)

# this function computes the E-step for all intervals in all time series parallely.
# the accummulated mmat and rvec are then used to solve the system, theta = mmat * rvec,
# to get the next iteration of theta (M-step). The function returns successfully if the
# absolute error goes below specified tolerance. The function returns unsuccessfully if the
# number of M-step iterations go beyond a threshold without reducing the error below tolerance.
def em(allx, allt, em_param, d_param):
    done = False
    numiter = 0

    while (done == False):
        numiter = numiter + 1
        print(numiter)
        mmat = np.zeros((prm.dim, prm.dof, prm.dof))
        rvec = np.zeros((prm.dim, prm.dof))
        
        ## this parallelization is for all time series observations in 1 go
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(delayed(mcmc2D)(allx, allt, em_param, d_param, path_index, step_index, dim_index) for path_index in range(allx.shape[0]) for step_index in range(allx.shape[1] - 1) for dim_index in range(allx.shape[2]))
            for res in results:
                mmat += res[0]
                rvec += res[1]
                print(", path index:", res[4], ", step index: ", res[5], ", dim index: ", res[6], ", AR burin:", res[2], ", AR sampling:", res[3])

        newtheta = np.linalg.solve(mmat, rvec)
        error = np.sum(np.abs(newtheta - theta))

        # if error is below tolerance, EM has converged
        if (error < em_param.tol):
            print("Finished successfully!")
            done = True

        # if number of iterations has crossed an iteration threshold, without
        # successfully redcuing the error below the error tolerance, then 
        # return unsuccessfully
        if (numiter > em_param.niter):
            print("Finished without reaching the tolerance")
            done = True

        d_param.theta = newtheta
        print(error)
        print(theta)

    return error, theta