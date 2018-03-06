import numpy as np
from joblib import Parallel, delayed

# this function defines our basis functions
# x must be a numpy array, a column vector of points
# (x = vector of points at which we seek to evaluate the basis functions)
# dof is the number of degrees of freedom, i.e., the number of basis functions
"""
def mypoly(x, dof):
    y = np.zeros((x.shape[0], dof))
    for i in range(x.shape[0]):
        for j in range(dof):
            H = poly.hermitenorm(j, monic = True)
            y[i,j] = H(x[i]) / np.sqrt(np.sqrt(2 * np.pi) * factorial(j))
    return y
"""

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
def hermite_basis(x, dof, dim):
    y = np.zeros((dim, dof))

    y[:, 0] = H0()

    index = 1
    for i in range(dim):
        y[:, index] = H1(x[i])
        y[:, index + 1] = H2(x[i])
        y[:, index + 2] = H3(x[i])
        index = 4
    
    index = 7
    for i in range(4):
        for j in range(3):
            y[:, index] = np.dot(y[:, i], y[:, (4 + j)])
            index += 1

    return y

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x, dof):
    y = np.zeros((dim, dof))

    y[:, 0] = 1

    index = 1
    for i in range(dim):
        y[:, index] = x[i]
        y[:, index + 1] = np.power(x[i], 2)
        y[:, index + 2] = np.power(x[i], 3)
        index = 4
    
    index = 7
    for i in range(4):
        for j in range(3):
            y[:, index] = np.dot(y[:, i], y[:, (4 + j)])
            index += 1

    return y

# drift function using "basis" functions defined by mypoly
# x must be a numpy array, the points at which the drift is to be evaluated
# theta must also be a numpy array, the coefficients of each "basis" function
# dof is implicitly calculated based on the first dimension of theta
def drift(x, theta):
    dof = theta.shape[0]
    dim = theta.shape[1]

    evaluated_basis = hermite_basis(x, dof, dim)
    out = np.zeros(dim)
    out[0] = np.sum(np.dot(evaluated_basis[0, :], theta[:, 0]))
    out[1] = np.sum(np.dot(evaluated_basis[1, :], theta[:, 1]))
    return out

def diffusion(g):
    return np.dot(g, np.random.standard_normal(2))

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
def createpaths(coef, g, numsteps, savesteps, h, ic, it):
    h12 = np.sqrt(h)
    numpaths = ic.shape[0]
    dim = ic.shape[1]

    x = np.zeros(( numpaths, (savesteps + 1), dim))
    t = np.zeros(( numpaths, (savesteps + 1), dim))

    x[:, 0, :] = ic
    t[:, 0, :] = it

    # for each time series, generate the matrix of size savesteps * dim
    # corresponding to one 2D time series
    for k in range(numpaths):
        # k-th initial condition to start off current x and t
        curx = ic[k, :]
        curt = it[k, :]
        j = 1
        for i in range(1, numsteps + 1):
            curx += drift(curx, coef) * h + diffusion(g) * h12
            curt += h
            if (i % (numsteps // savesteps) == 0):
                x[k, j, :] = curx
                t[k, j, :] = curt
                j += 1

    return x, t

# creates brownian bridge interpolations for given start and end
# time point t and value x.
def brownianbridge(g, xin, tin, n, index):
    h = (tin[index + 1] - tin[index]) / n
    tvec = tin[index] + (1+np.arange(n))*h
    h12 = np.sqrt(h)

    # W ~ N(0, sqrt(h)*g)
    wincs = np.random.normal(scale=h12*g, size=n)
    w = np.cumsum(wincs)

    # TODO: insert brownian bridge formula
    bridge = xin[index] + w - ((tvec - tin[index])/(tin[index + 1]-tin[index]))*(w[n-1] + xin[index] - xin[index + 1])
    
    # concatenate the starting point to the bridge
    tvec = np.concatenate((tin[[index]], tvec))
    bridge = np.concatenate((xin[[index]],bridge))
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
    mmat = np.zeros((d_param.dim, d_param.dof, d_param.dof))
    rvec = np.zeros((d_param.dim, d_param.dof))

    # one time series at a time
    x = allx[:, j]
    t = allt[:, j]
    
    samples = np.zeros(numsubintervals)
    _, xcur = brownianbridge(g,x,t,numsubintervals,i)
    oldlik = girsanov(g=g, path=xcur, dt=h, theta=theta)
    arburn = np.zeros(burninpaths)
    for jj in range(burninpaths):
        _, prop = brownianbridge(g,x,t,numsubintervals,i)
        proplik = girsanov(g=g, path=prop, dt=h, theta=theta)
        rho = np.exp(proplik - oldlik)
        if (rho > np.random.uniform()):
            xcur = prop
            oldlik = proplik
            arburn[jj] = 1
    meanBurnin = np.mean(arburn)
    
    # for each path being sampled (r = 0 to r = R)
    arsamp = np.zeros(numpaths)
    for jj in range(numpaths):
        _, prop = brownianbridge(g,x,t,numsubintervals,i)
        proplik = girsanov(g=g, path=prop, dt=h, theta=theta)
        rho = np.exp(proplik - oldlik)
        if (rho > np.random.uniform()):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1
        samples = xcur
        pp = hermite_basis(samples[:(-1)], dof)
        mmat = mmat + h * np.matmul(pp.T, pp) / numpaths
        rvec = rvec + np.matmul((np.diff(samples)).T, pp) / numpaths    
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
        mmat = np.zeros((d_param.dim, d_param.dof, d_param.dof))
        rvec = np.zeros((d_param.dim, d_param.dof))
        
        ## this parallelization is for all time series observations in 1 go
        with Parallel(n_jobs=-1) as parallel:
            results = parallel(delayed(mcmc2D)(allx, allt, em_param, d_param, path_index, step_index, dim_index) for path_index in range(allx.shape[0]) for step_index in range(allx.shape[1] - 1) for dim_index in range(allx.shape[2]))
            for res in results:
                mmat += res[0]
                rvec += res[1]
                print(", path index:", res[4], "step index: ", res[5] ", dim index: ", res[6] ", AR burin:", res[2], ", AR sampling:", res[3])

        newtheta = np.linalg.solve(mmat, rvec)
        error = np.sum(np.abs(newtheta - theta))

        # if error is below tolerance, EM has converged
        if (error < mytol):
            print("Finished successfully!")
            done = True

        # if number of iterations has crossed an iteration threshold, without
        # successfully redcuing the error below the error tolerance, then 
        # return unsuccessfully
        if (numiter > niter):
            print("Finished without reaching the tolerance")
            done = True

        theta = newtheta
        print(error)
        print(theta)

    return error, theta