import numpy as np

# function that creates a Brownian bridge in R^d
# dX_t = diag(gvec) * dW_t
# xin should have dimensions 2 x d
# tin should have dimension 2
# the returned bridge will take n steps to go from xin[0,:] to xin[1,:]
# n steps are returned
def brownianbridgeM(gvec, xin, tin, n):
    h = (tin[1]-tin[0])/n
    tvec = tin[0] + (1+np.arange(n))*h
    h12 = np.sqrt(h)
    wincs = np.random.multivariate_normal(mean=np.zeros(len(gvec)),
                                          cov=h*np.diag(np.square(gvec)),
                                          size=n)
    w = np.cumsum(wincs,axis=0).T
    bridge = xin[0,:,None] + w
    bridge -= ((tvec - tin[0])/(tin[1]-tin[0]))*(w[:,n-1,None] + xin[0,:,None] - xin[1,:,None])
    return tvec, bridge

# using girsanov's theorem, this function
# computes the LOG likelihood of the "path", with time step dt,
# relative to a Brownian bridge with diffusion matrix diag(gvec)
def girsanovM(gvec, path, dt):
    b = drift(path)
    u = np.dot(np.diag(np.power(gvec,-2)), b)
    int1 = np.tensordot(u[:,:-1], np.diff(path,axis=1))
    # int2 = np.tensordot(u, b)
    u2 = np.einsum('ij,ji->i',u.T,b)
    int2 = np.sum(0.5*(u2[1:] + u2[:-1]))*dt
    r = int1 - 0.5*int2
    return r

# metropolis diffusion bridge sampler
# burnin = number of metropolis burn-in steps
# numsteps = number of metropolis steps POST-burn-in, i.e.,
#            how many diffusion bridge path samples do you want?
# gvec, xin and tin are just like for the Brownian bridge function above
# n = number of time steps each diffusion bridge path should take

def diffusionbridge(burnin, numsteps, gvec, xin, tin, n):

    # dimension of diffusion bridge
    d = len(gvec)

    # time step 
    h = (tin[1]-tin[0])/n

    # storing everything, which is not strictly necessary to compute
    # expected values!  (just doing it here for generality's sake)
    samples = np.zeros((numsteps, d, n))

    # create initial path and likelihood
    tout, xcur = brownianbridgeM(gvec,x,t,n)
    oldlik = girsanovM(gvec, xcur, h)

    # to track accept/reject rate during burn-in and post-burn-in
    arburn = np.zeros(burnin)
    arsamp = np.zeros(numsteps)

    # burn-in loop
    for jj in range(burnin):
        _, prop = brownianbridgeM(gvec,x,t,n)
        proplik = girsanovM(gvec, prop, h)
        rho = proplik - oldlik
        if (rho >= np.log(np.random.uniform())):
            xcur = prop
            oldlik = proplik
            arburn[jj] = 1
        
    # post-burn-in loop; now we save samples
    for jj in range(numsteps):
        _, prop = brownianbridgeM(gvec,x,t,n)
        proplik = girsanovM(gvec, prop, h)
        rho = proplik - oldlik
        if (rho >= np.log(np.random.uniform())):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1
        samples[jj,:] = xcur

    return samples, np.mean(arburn), np.mean(arsamp)


#
# test this code
#

# define drift function
def drift(x):
    f = np.zeros(x.shape)
    f[0,:] = x[1,:]
    f[1,:] = -np.sin(x[0,:])
    return f

# we assume the diffusion matrix is diag(gamma)
# hence the entries of gamma should be strictly positive
gamma = np.array([.2,.25])

x = np.array([[0.5,0.6],[0.3,0.2]])
t = np.array([0., 0.75])

test, arb, ars = diffusionbridge(100, 50, gamma, x, t, 100)

