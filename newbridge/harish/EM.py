import numpy as np
import newbridge as nb
from joblib import Parallel

# load data
import pickle
with open('nem.pkl','rb') as f:
    allx, allt = pickle.load(f)

print(allx.shape)
print(allt.shape)

# initial guess for theta
theta = np.array([1.5, 1., 0., -0.25])
dof = theta.shape[0]

# true diffusion coefficient
g = 0.5

numsubintervals = 10
h = (allt[1,0] - allt[0,0])/numsubintervals
numpaths = 1000
burninpaths = 10

def mcmc(burninpaths, numpaths, g, x, t, numsubintervals, i, h, theta, dof):
    mmat = np.zeros((dof, dof))
    rvec = np.zeros(dof)
    samples = np.zeros(numsubintervals)
    _, xcur = nb.brownianbridge(g,x,t,numsubintervals,i)
    oldlik = nb.girsanov(g=g, path=xcur, dt=h, theta=theta)
    arburn = np.zeros(burninpaths)
    for jj in range(burninpaths):
        _, prop = nb.brownianbridge(g,x,t,numsubintervals,i)
        proplik = nb.girsanov(g=g, path=prop, dt=h, theta=theta)
        rho = np.exp(proplik - oldlik)
        if (rho > np.random.uniform()):
            xcur = prop
            oldlik = proplik
            arburn[jj] = 1
    meanBurnin = np.mean(arburn)
    
    # for each path being sampled (r = 0 to r = R)
    arsamp = np.zeros(numpaths)
    for jj in range(numpaths):
        _, prop = nb.brownianbridge(g,x,t,numsubintervals,i)
        proplik = nb.girsanov(g=g, path=prop, dt=h, theta=theta)
        rho = np.exp(proplik - oldlik)
        if (rho > np.random.uniform()):
            xcur = prop
            oldlik = proplik
            arsamp[jj] = 1
        samples = xcur
        pp = nb.mypoly(samples[:(-1)], dof)
        mmat = mmat + h * np.matmul(pp.T, pp) / numpaths
        rvec = rvec + np.matmul((np.diff(samples)).T, pp) / numpaths    
    meanSample = np.mean(arsamp)
    
    return (mmat, rvec, meanBurnin, meanSample)

done = False
mytol = 1e-3
numiter = 0
while (done == False):
    numiter = numiter + 1
    print(numiter)
    mmat = np.zeros((dof, dof))
    rvec = np.zeros(dof)
    
    ## each column of x and t forms a time series observation
    ## this parallelization is for each time series
    # for wp in range(allx.shape[1]):
    #     x = allx[:,wp]
    #     t = allt[:,wp]
    #     with Parallel(n_jobs=-1) as parallel:
    #         results = parallel(delayed(mcmc)(burninpaths, numpaths, g, x, t, numsubintervals, i, h, theta, dof) 
    #                         for i in range(x.shape[0] - 1))
    #         for res in results:
    #             mmat += res[0]
    #             rvec += res[1]
    #             print("Acceptance rate during burn-in:", res[2])
    #             print("Acceptance rate post burn-in:", res[3])

    ## this paralleization is for all time series observations in 1 go
    with Parallel(n_jobs=-1) as parallel:
    results = parallel(delayed(mcmc)(burninpaths, numpaths, g, allx, allt, numsubintervals, i, j, h, theta, dof) 
                    for (i, j) in zip(range(allx.shape[0] - 1), range(allx.shape[1])))
    for res in results:
        mmat += res[0]
        rvec += res[1]
        print("Acceptance rate during burn-in:", res[2])
        print("Acceptance rate post burn-in:", res[3])

    newtheta = np.linalg.solve(mmat, rvec)
    check = np.sum(np.abs(newtheta - theta))
    if (check < mytol):
        print("finished!")
        print(check)
        print(theta)
        done = True
    theta = newtheta
    print(check)
    print(theta)