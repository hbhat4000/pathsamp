import numpy as np
import polynomial_functions as pfn
import scipy.special

def find_dof(degree, dim):
	return int(scipy.special.binom(degree + dim - 1, dim))

def index_mapping(terms):
    index = 0
    index_map = {}

    for d in range(0, terms):
        for l in range(0, d + 1):
            for k in range(0, d + 1):
                for j in range(0, d + 1):
                    for i in range(0, d + 1):
                        if (i + j + k + l == d):
                            index_set = (i, j, k, l)
                            index_map[index_set] = index
                            index += 1

    return index_map

def h2o_simple_transformation(terms):
    mat = np.zeros((terms, terms))
    mat[0, 0] = 0.63161877774606470129
    mat[1, 1] = 0.63161877774606470129
    mat[2, 2] = 0.44662192086900116570
    mat[0, 2] = -mat[2, 2]
    mat[3, 3] = 0.25785728623970555997
    mat[1, 3] = -3 * mat[3, 3]

    return mat

# for the dim-dimensional, terms-hermite terms case, creating the transformation matrix for
# any index mapping provided
def h2o_transformation_matrix(dim, dof, index_map, h2o_mat):
    transformation = np.full((dof, dof), 1.)

    for row_index in index_map:
        for col_index in index_map:
            for d in range(dim):
                transformation[index_map[row_index], index_map[col_index]] *= h2o_mat[row_index[d], col_index[d]]
                
    return transformation

num_hermite_terms = 4
dim = 4
dof = find_dof(num_hermite_terms, dim)
index_map = index_mapping(num_hermite_terms)
h2o_mat = h2o_simple_transformation(num_hermite_terms)
transformation = h2o_transformation_matrix(dim, dof, index_map, h2o_mat)

class em:
	def __init__(self, tol, burninpaths, mcmcpaths, numsubintervals, niter, dt):
		self.tol = tol	# tolerance for error in the theta value
		self.burninpaths = burninpaths 	# burnin paths for mcmc
		self.mcmcpaths = mcmcpaths	# sampled paths for mcmc
		self.numsubintervals = numsubintervals	# number of sub intervals in each interval [x_i, x_{i+1}] for the Brownian bridge
		self.niter = niter	# threshold for number of EM iterations, after which EM returns unsuccessfully
		self.h = dt / numsubintervals	# time step for EM

class data:
	def __init__(self, theta, gvec):
		self.theta = theta
		self.gvec = gvec

class euler_maruyama:
	def __init__(self, numsteps, savesteps, ft, ic, it, numpaths):
		self.numsteps = numsteps
		self.savesteps = savesteps
		self.ft = ft
		self.h = ft / numsteps
		self.ic = ic
		self.it = it
		self.numpaths = numpaths

class system:
	def __init__(self, kvec, mvec, gvec):
		self.kvec = kvec
		self.mvec = mvec
		self.gvec = gvec

class theta_transformations:
    def __init__(self, theta, theta_type=None):
        if theta_type is 'ordinary':
            self.ordinary = theta
            self.hermite = nb.ordinary_to_hermite(theta)
            self.sparse_ordinary = nb.theta_sparsity(self.ordinary)
            self.sparse_hermite = nb.theta_sparsity(self.hermite)
        if theta_type is 'hermite':
            self.ordinary = nb.hermite_to_ordinary(theta)
            self.hermite = theta
            self.sparse_ordinary = nb.theta_sparsity(self.ordinary)
            self.sparse_hermite = nb.theta_sparsity(self.hermite)

