import numpy as np
import parameters as prm
from joblib import Parallel, delayed

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    theta = np.zeros((x.shape[0], prm.dof))
    index = 0

    for d in range(0, prm.num_hermite_terms):
        for i in range(0, d + 1):
            if (i == d):
                # print("d", d, "i", i, "index", index)
                theta[:, index] = np.power(x[:, 0], i)
                index += 1

    return theta

def H(degree, x):
    switcher = {
        0: 0.63161877774606470129,
        1: 0.63161877774606470129 * x,
        2: 0.44662192086900116570 * (np.power(x, 2) - 1),
        3: 0.25785728623970555997 * (np.power(x, 3) - 3 * x),
        4: 0.12892864311985277998 * (np.power(x, 4) - 6 * np.power(x, 2) + 3),
    }
    return switcher.get(degree, "Polynomial degree exceeded")

# this function defines our hermite basis functions
# x must be a numpy array, a column vector of points
# (x = vector of points at which we seek to evaluate the basis functions)
# dof is the number of degrees of freedom, i.e., the number of basis functions.

# TODO : currently this loop has to be written separately for varying dimensions.
def hermite_basis(x):
    theta = np.zeros((x.shape[0], prm.dof))
    index = 0

    for d in range(0, prm.num_hermite_terms):
        for i in range(0, d + 1):
            if (i == d):
                # print("d", d, "i", i, "index", index)
                theta[:, index] = H(i, x[:, 0])
                index += 1

    return theta

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
def girsanov(d_param, em_param, path, tdiff):
	# path is of size (numsubintervals + 1, dim)
	# b is of size (numsubintervals + 1, dim)
	b = drift(d_param, path)
	u = np.dot(np.diag(np.power(d_param.gvec, -2)), b.T).T
	int1 = np.tensordot(u[:-1, :], np.diff(path, axis = 0))
	u2 = np.einsum('ij,ji->i', u.T, b)
	int2 = np.sum(0.5 * (u2[1:] + u2[:-1])) * (tdiff)
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

    # one time series, one interval, all dimensions at a time
    x = allx[path_index, step_index:(step_index + 2), :]
    t = allt[path_index, step_index:(step_index + 2)]
    tdiff = (t[1] - t[0]) / em_param.numsubintervals

    samples = np.zeros((em_param.numsubintervals, prm.dim))
    _, xcur = brownianbridge(d_param, em_param, x, t)
    oldlik = girsanov(d_param, em_param, xcur, tdiff)

    arburn = np.zeros(em_param.burninpaths)
    for jj in range(em_param.burninpaths):
        _, prop = brownianbridge(d_param, em_param, x, t)
        proplik = girsanov(d_param, em_param, prop, tdiff)

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
        proplik = girsanov(d_param, em_param, prop, tdiff)

        rho = proplik - oldlik
        if (rho > np.log(np.random.uniform())):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1

        samples = xcur
        pp = hermite_basis(samples[:(-1)])
        mmat = mmat + tdiff * np.matmul(pp.T, pp) / em_param.mcmcpaths
        rvec = rvec + np.matmul((np.diff(samples, axis = 0)).T, pp) / em_param.mcmcpaths   
    meanSample = np.mean(arsamp)
    
    return (mmat, rvec, meanBurnin, meanSample, path_index, step_index)

def index_mapping():
    index = 0
    index_map = {}

    for d in range(0, prm.num_hermite_terms):
        for i in range(0, d + 1):
            if (i == d):
                index_set = (i)
                index_map[index_set] = index
                index += 1

    return index_map

def h2o_simple_transformation():
    mat = np.zeros((prm.num_hermite_terms, prm.num_hermite_terms))
    mat[0, 0] = 0.63161877774606470129
    mat[1, 1] = 0.63161877774606470129
    mat[2, 2] = 0.44662192086900116570
    mat[0, 2] = -mat[2, 2]
    mat[3, 3] = 0.25785728623970555997
    mat[1, 3] = -3 * mat[3, 3]

    return mat

def h2o_transformation_matrix():
    transformation = np.zeros((prm.dof, prm.dof))
    index_map = index_mapping()
    index = 0

    mat = h2o_simple_transformation()

    for d in range(0, prm.num_hermite_terms):
        for i in range(0, d + 1):
            if (i == d):
                transformation[index, index] = mat[i, i]
                if (i >= 2):
                    new_index_set = (i - 2)
                    new_index = index_map[new_index_set]
                    transformation[new_index, index] = mat[i - 2, i]

                index += 1

    return transformation

def hermite_to_ordinary(theta):
    transformation = h2o_transformation_matrix() 
    ordinary_theta = np.matmul(transformation, theta)
    return ordinary_theta

def ordinary_to_hermite(theta):
    transformation = np.linalg.inv(h2o_transformation_matrix())
    hermite_theta = np.matmul(transformation, theta)
    return hermite_theta

def norm_error(true_theta, estimated_theta):
    errors = []
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.ordinary - estimated_theta.ordinary), 2))))
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.hermite - estimated_theta.hermite), 2))))
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.sparse_ordinary - estimated_theta.sparse_ordinary), 2))))
    errors.append(np.sqrt(np.sum(np.power(np.abs(true_theta.sparse_hermite - estimated_theta.sparse_hermite), 2))))
    return errors


def theta_sparsity(theta):
    threshold = 0.1 * np.max(np.abs(theta))
    theta[np.abs(theta) < threshold] = 0.
    return theta

def residual(allx, allt, em_param, d_param, path_index, step_index):
    gammavecsq = np.zeros((prm.dim))

    # one time series, one interval, all dimensions at a time
    x = allx[path_index, step_index:(step_index + 2), :]
    t = allt[path_index, step_index:(step_index + 2)]
    tdiff = (t[1] - t[0]) / em_param.numsubintervals

    samples = np.zeros((em_param.numsubintervals, prm.dim))
    _, xcur = brownianbridge(d_param, em_param, x, t)
    oldlik = girsanov(d_param, em_param, xcur, tdiff)

    arburn = np.zeros(em_param.burninpaths)
    for jj in range(em_param.burninpaths):
        _, prop = brownianbridge(d_param, em_param, x, t)
        proplik = girsanov(d_param, em_param, prop, tdiff)

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
        proplik = girsanov(d_param, em_param, prop, tdiff)

        rho = proplik - oldlik
        if (rho > np.log(np.random.uniform())):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1

        samples = xcur
        pp = hermite_basis(samples[:(-1)])
        gammavecsq = gammavecsq + np.sum(np.square(np.diff(samples, axis = 0) - tdiff * np.matmul(pp, d_param.theta)), axis = 0) / (tdiff * em_param.mcmcpaths * (em_param.numsubintervals * (allx.shape[1] - 1) + 1))
    
    meanSample = np.mean(arsamp)

    return (gammavecsq, meanBurnin, meanSample, path_index, step_index)

def infer_noise(allx, allt, em_param, d_param):
    gammavec = np.zeros((prm.dim))
    
    with Parallel(n_jobs=-1) as parallel:
        results = parallel(delayed(residual)(allx, allt, em_param, d_param, path_index, step_index)
            for path_index in range(allx.shape[0]) for step_index in range(allx.shape[1] - 1))
        for res in results:
            gammavec += res[0]
    return gammavec

# this function computes the E-step for all intervals in all time series parallely.
# the accummulated mmat and rvec are then used to solve the system, theta = mmat * rvec,
# to get the next iteration of theta (M-step). The function returns successfully if the
# absolute error goes below specified tolerance. The function returns unsuccessfully if the
# number of M-step iterations go beyond a threshold without reducing the error below tolerance.
def em(allx, allt, em_param, d_param):
    done = False
    numiter = 0
    error_list = []
    theta_list = []

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

        # relative error
        error = np.sum(np.abs(newtheta - d_param.theta)) / np.sum(np.abs(d_param.theta))

        # inducing sparsity in the Hermite space
        newtheta[np.abs(newtheta) < 0.01] = 0.
        d_param.theta = newtheta

        error_list.append(error)
        theta_list.append(d_param.theta)
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

        print(error)
        print(d_param.theta)

    return error_list, theta_list
