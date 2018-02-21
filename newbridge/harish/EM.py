import numpy as np
import newbridge as nb

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

done = False
mytol = 1e-3
numiter = 0
while (done == False):
    numiter = numiter + 1
    print(numiter)
    mmat = np.zeros((dof, dof))
    rvec = np.zeros(dof)
    
    # for each interval of observed value (x_0 to x_n)
    for wp in range(allx.shape[1]):
        x = allx[:,wp]
        t = allt[:,wp]
        for i in range(x.shape[0]-1):
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
            print("Acceptance rate during burn-in:", np.mean(arburn))

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
            print("Acceptance rate post burn-in:", np.mean(arsamp))

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



