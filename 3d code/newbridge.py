import numpy as np
import parameters as prm
from joblib import Parallel, delayed

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    y = np.zeros((x.shape[0], prm.dof))

    # constant polynomial
    index = 0
    y[:, index] = 1.

    # 1st order polynomial: dim * (degree - 1)
    for i in range(prm.dim):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = np.power(x[:, i], j)

    if (index == (prm.dof - 1)):
        return y

    # 2nd order polynomial: dim * (degree - 1)^2
    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = y[:, i] * y[:, (prm.polynomial_degree - 1 + j)]

    if (index == (prm.dof - 1)):
        return y

    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = y[:, i] * y[:, (2 * (prm.polynomial_degree - 1) + j)]

    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = y[:, (prm.polynomial_degree - 1 + i)] * y[:, (2 * (prm.polynomial_degree - 1) + j)]

    # 3rd polynomial: degree^3
    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            for k in range(1, prm.polynomial_degree):
                index += 1
                y[:, index] = y[:, i] * y[:, (prm.polynomial_degree - 1 + j)] * y[:, (2 * (prm.polynomial_degree - 1) + k)]

    if (index == (prm.dof - 1)):
        return y

def H(degree, x):
    switcher = {
        0: 0.63161878,
        1: 0.63161878 * x,
        2: 0.44662192 * (np.power(x, 2) - 1),
        3: 0.25785728 * (np.power(x, 3) - 3 * x),
        4: 0.12892864 * (np.power(x, 4) - 6 * np.power(x, 2) + 3),
        5: 0.05765864 * (np.power(x, 5) - 10 * np.power(x, 3) + 15 * x)
    }
    return switcher.get(degree, "Polynomial degree exceeded")

# this function defines our hermite basis functions
# x must be a numpy array, a column vector of points
# (x = vector of points at which we seek to evaluate the basis functions)
# dof is the number of degrees of freedom, i.e., the number of basis functions
def hermite_basis(x):
    y = np.zeros((x.shape[0], prm.dof))

    # constant polynomial
    index = 0
    y[:, index] = H(index, x)

    # 1st order polynomial: dim * (degree - 1)
    for i in range(prm.dim):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = H(j, x[:, i])

    if (index == (prm.dof - 1)):
        return y

    # 2nd order polynomial: dim * (degree - 1)^2
    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = y[:, i] * y[:, (prm.polynomial_degree - 1 + j)]

    if (index == (prm.dof - 1)):
        return y

    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = y[:, i] * y[:, (2 * (prm.polynomial_degree - 1) + j)]

    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            index += 1
            y[:, index] = y[:, (prm.polynomial_degree - 1 + i)] * y[:, (2 * (prm.polynomial_degree - 1) + j)]

    # 3rd polynomial: degree^3
    for i in range(1, prm.polynomial_degree):
        for j in range(1, prm.polynomial_degree):
            for k in range(1, prm.polynomial_degree):
                index += 1
                y[:, index] = y[:, i] * y[:, (prm.polynomial_degree - 1 + j)] * y[:, (2 * (prm.polynomial_degree - 1) + k)]

    if (index == (prm.dof - 1)):
        return y

# drift function using "basis" functions defined by mypoly
# x must be a numpy array, the points at which the drift is to be evaluated
# theta must also be a numpy array, the coefficients of each "basis" function
# dof is implicitly calculated based on the first dimension of theta
def drift(d_param, x):
    evaluated_basis = np.zeros((prm.dim, x.shape[0], prm.dof))
    out = np.zeros((x.shape[0], prm.dim))

    # one dimension at a time, each row of x gets mapped to the hermite basis.
    # both dimensions of x are passed to the hermite function since the hermite
    # functions depend on all dimensions of x.
    for i in range(prm.dim):
        evaluated_basis[i, :, :] = hermite_basis(x)
        out[:, i] = np.sum(np.dot(evaluated_basis[i, :, :], d_param.theta[:, i]))

    return out

def diffusion(d_param):
    return np.dot(d_param.gvec, np.random.standard_normal(prm.dim))

# create sample paths! 

# this function creates a bunch of Euler-Maruyama paths from an array
# of initial conditions
def createpaths(d_param, euler_param):
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
            curx += drift(d_param, curx) * euler_param.h + diffusion(d_param) * h12
            curx_without_noise += drift(d_param, curx_without_noise) * euler_param.h
            curt += euler_param.h
            if (i % (euler_param.numsteps // euler_param.savesteps) == 0):
                x[k, j, :] = curx
                x_without_noise[k, j, :] = curx_without_noise
                t[k, j] = curt
                j += 1

    return x, t, x_without_noise

# creates brownian bridge interpolations for given start and end
# time point t and value x.
def brownianbridge(d_param, em_param, xin, tin):
    h = (tin[1] - tin[0]) / em_param.numsubintervals
    tvec = tin[0] + (1 + np.arange(em_param.numsubintervals)) * h
    h12 = np.sqrt(h)

    # W ~ N(0, sqrt(h)*g)
    wincs = np.random.multivariate_normal(mean = np.zeros(prm.dim), 
        cov = h * np.diag(np.square(d_param.gvec)), 
        size = em_param.numsubintervals)
    w = np.cumsum(wincs, axis = 0).T

    bridge = xin[0, :, None] + w
    bridge -= ((tvec - tin[0])/(tin[1]-tin[0]))*(w[:, em_param.numsubintervals - 1, None] + xin[0, :, None] - xin[1, :, None])
    
    # concatenate the starting point to the bridge
    # tvec.shape is (11, ) that is (numsubintervals + 1, )
    # bridge.shape is (11, 2) that is (numsubintervals + 1, dim)
    tvec = np.concatenate((tin[[0]], tvec)).T
    bridge = np.concatenate((xin[0, :, None],bridge), axis = 1).T
    return tvec, bridge

# Girsanov likelihood is computed using # TODO: insert reference to the paper
def girsanov(d_param, em_param, path):
	# path is of size (numsubintervals + 1, dim)
	# b is of size (numsubintervals + 1, dim)
	b = drift(d_param, path)
	u = np.dot(np.diag(np.power(d_param.gvec, -2)), b.T).T
	int1 = np.tensordot(u[:-1, :], np.diff(path, axis = 0))
	u2 = np.einsum('ij,ji->i', u.T, b)
	int2 = np.sum(0.5 * (u2[1:] + u2[:-1])) * em_param.h
	r = int1 - 0.5 * int2
	return r

# this function computes MCMC steps for i-th interval of the j-th time series
# using Brownian bridge. The accept-reject step is computed using the Girsanov
# likelihood function. First burnin steps are rejected and the next numsteps
# are used to compute the mmat and rvec (E step) which are used to solve the system of 
# equations producing the next iteration of theta (M step).
def mcmc(allx, allt, d_param, em_param, path_index, step_index):
    mmat = np.zeros((prm.dim, prm.dof, prm.dof))
    rvec = np.zeros((prm.dim, prm.dof))

    # one time series, one interval, one dimension at a time
    x = allx[path_index, step_index:(step_index + 2), :]
    t = allt[path_index, step_index:(step_index + 2)]
    
    samples = np.zeros((em_param.numsubintervals, prm.dim))
    _, xcur = brownianbridge(d_param, em_param, x, t)
    oldlik = girsanov(d_param, em_param, xcur)
    arburn = np.zeros(em_param.burninpaths)
    for jj in range(em_param.burninpaths):
        _, prop = brownianbridge(d_param, em_param, x, t)
        proplik = girsanov(d_param, em_param, prop)
        rho = proplik - oldlik
        if (rho > np.log(np.random.uniform())):
            xcur = prop
            oldlik = proplik
            arburn[jj] = 1
    meanBurnin = np.mean(arburn)
    
    # for each path being sampled (r = 0 to r = R)
    arsamp = np.zeros(em_param.mcmcpaths)
    for jj in range(em_param.mcmcpaths):
        _, prop = brownianbridge(d_param, em_param, x, t)
        proplik = girsanov(d_param, em_param, prop)
        rho = proplik - oldlik
        if (rho > np.log(np.random.uniform())):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1
        samples = xcur
        # pp = hermite_basis(samples[:(-1)])
        pp = hermite_basis(samples[:(-1)])
        mmat = mmat + em_param.h * np.matmul(pp.T, pp) / em_param.mcmcpaths
        rvec = rvec + np.matmul((np.diff(samples, axis = 0)).T, pp) / em_param.mcmcpaths   
    meanSample = np.mean(arsamp)
    
    return (mmat, rvec, meanBurnin, meanSample, path_index, step_index)

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
            results = parallel(delayed(mcmc)(allx, allt, d_param, em_param, path_index, step_index) 
                for path_index in range(allx.shape[0]) for step_index in range(allx.shape[1] - 1))
            for res in results:
                mmat += res[0]
                rvec += res[1]
                print("path index:", res[4], ", step index: ", res[5], ", AR burin:", res[2], ", AR sampling:", res[3])

        newtheta = np.linalg.solve(mmat, rvec).T
        error = np.sum(np.abs(newtheta - d_param.theta)) / np.sum(np.abs(d_param.theta))

        # if a threshold is applied to theta
        # newtheta[np.abs(newtheta) < 0.5] = 0.

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
        print(d_param.theta)

    return error, d_param
