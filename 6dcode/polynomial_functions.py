import numpy as np
import parameters as prm

# defines polynomial basis functions {1, x, x^2, x^3}
def polynomial_basis(x):
    theta = np.full((x.shape[0], prm.dof), 1.)
    index_map = index_mapping()

    for index in index_map:
        for d in range(prm.dim):
            theta[:, index_map[index]] *= np.power(x[:, d], index[d])

    return theta

def H(x, degree):
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
def hermite_basis(x):
    theta = np.full((x.shape[0], prm.dof), 1.)
    index_map = index_mapping()

    Hcached = np.zeros((x.shape[0], x.shape[1], prm.num_hermite_terms))
    for d in range(prm.num_hermite_terms):
        Hcached[:, :, d] = H(x, d)

    for index in index_map:
        for d in range(prm.dim):
            theta[:, index_map[index]] *= Hcached[:, d, index[d]]

    return theta

def index_mapping():
    index = 0
    index_map = {}

    for d in range(0, prm.num_hermite_terms):
        for n in range(0, d + 1):
            for m in range(0, d + 1):
                for l in range(0, d + 1):
                    for k in range(0, d + 1):
                        for j in range(0, d + 1):
                            for i in range(0, d + 1):
                                if (i + j + k + l + m + n == d):
                                    index_set = (i, j, k, l, m, n)
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

# for the dim-dimensional, terms-hermite terms case, creating the transformation matrix for
# any index mapping provided
def h2o_transformation_matrix():
    transformation = np.full((prm.dof, prm.dof), 1.)
    index_map = index_mapping()
    mat = h2o_simple_transformation()

    for row_index in index_map:
        for col_index in index_map:
            for d in range(prm.dim):
                transformation[index_map[row_index], index_map[col_index]] *= mat[row_index[d], col_index[d]]
                
    return transformation

# converting theta is hermite space to ordianry space using the transformation matrix
def hermite_to_ordinary(theta):
    transformation = h2o_transformation_matrix() 
    ordinary_theta = np.matmul(transformation, theta)
    return ordinary_theta

# converting theta is ordinary space to hermite space using the inverse of transformation matrix
def ordinary_to_hermite(theta):
    transformation = np.linalg.inv(h2o_transformation_matrix())
    hermite_theta = np.matmul(transformation, theta)
    return hermite_theta

# hard thresholding for theta using a threshold
def theta_sparsity(theta, threshold):
    relative_threshold = threshold * np.max(np.abs(theta))
    theta[np.abs(theta) < relative_threshold] = 0.
    return theta

# computing regression and classification errors between true and estimated theta
def compute_errors(true, estimated, threshold):
    errors = []
    estimated = theta_sparsity(estimated, threshold)
    true = theta_sparsity(true, threshold)

    # regression metric
    # L1 norm
    errors.append(np.sum(np.abs(true - estimated)))

    # L2 norm
    errors.append(np.sqrt(np.sum(np.power(true - estimated, 2))))

    # classification metric, P = value is zero, N = value is non-zero
    # true positive => true was zero and estimated was zero
    TP = np.sum(np.logical_and(true == 0., estimated == 0.))
    # true negative => true was non-zero and estimated was non-zero
    TN = np.sum(np.logical_and(true != 0., estimated != 0.))
    # false positive => true was non-zero and estimated was zero
    FP = np.sum(np.logical_and(true != 0., estimated == 0.))
    # false negative => true was zero and estimated was non-zero
    FN = np.sum(np.logical_and(true == 0., estimated != 0.))

    # precision = true positives / total estimated positives {TP / (TP + FP)}
    errors.append(TP / (TP + FP))

    # recall = true positives / total true positives {TP / (TP + FN)}
    errors.append(TP / (TP + FN))
        
    # accuracy = total true / total predictions {(TP + TN) / (TP + TN + FP + FN)}
    errors.append((TP + TN) / (TP + TN + FP + FN))

    # F1 score = 2*precision*recall / (precision + recall)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    errors.append(2 * precision * recall / (precision + recall))

    return errors
