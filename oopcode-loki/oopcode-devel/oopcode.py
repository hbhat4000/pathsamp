import numpy as np
import multiprocessing
import scipy.special
import scipy.integrate
import hermite
# from autograd import elementwise_grad, jacobian
# import matplotlib.pyplot as plt
from functools import lru_cache

# tensor product of Hermite polynomials
class Hermite:

    # maxdeg = maximum degree of 1-d Hermite poly
    # dim = dimension, i.e., the d in R^d
    def __init__(self, maxdeg, dim):
        if (maxdeg < 0):
            raise ValueError("Maximum degree must be at least 0.")
        if (dim < 1):
            raise ValueError("Dimension must be at least 1.")
        self.maxdeg = maxdeg + 1
        self.dim = dim
        self.dof = int(scipy.special.binom(self.maxdeg + dim - 1, dim))
        self.digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.index_map = self.index_mapping()
        self.ns = np.zeros(9)
        for ii in range(9):
            self.ns[ii] = 1.0/np.sqrt(np.sqrt(2 * np.pi) * scipy.special.factorial(ii))
        self.seth2omat()
        self.settransmat()

    def padder(self, x):
        if len(x) < self.dim:
            return '0' * (self.dim - len(x)) + x
        else:
            return x

    def int2str(self, x):
        if x < 0:
            return "-" + self.int2str(-x)
        return ("" if x < self.maxdeg else self.int2str(x // self.maxdeg)) \
               + self.digits[x % self.maxdeg]

    def padint2str(self, x):
        return self.padder(self.int2str(x))

    # index mapping
    def index_mapping(self):
        index = 0
        index_map = {}
        allpos = list(map(self.padint2str, range(self.maxdeg ** self.dim)))
        for d in range(self.maxdeg):
            for s in allpos:
                y = list(map(int, s))[::-1]
                if sum(y) == d:
                    index_map[tuple(y)] = index
                    index += 1

        return index_map

    # 1-d Hermite polys
    def H(self, x, degree):
        switcher = {
            0: self.ns[0],
            1: self.ns[1] * x,
            2: self.ns[2] * (x**2 - 1),
            3: self.ns[3] * (x**3 - 3 * x),
            4: self.ns[4] * (x**4 - 6 * x**2 + 3),
            5: (x**5 - 10 * x**3 + 15 * x) * self.ns[5],
            6: (x**6 - 15 * x**4 + 45 * x**2 - 15) * self.ns[6],
            7: (x**7 - 21 * x**5 + 105 * x**3 - 105 * x) * self.ns[7],
            8: (x**8 - 28 * x**6 + 210 * x**4 - 420 * x**2 + 105) * self.ns[8]
        }
        return switcher.get(degree, "Polynomial degree exceeded")


    def seth2omat(self):
        self.h2omat = np.zeros((9,9))
        for ii in range(9):
            self.h2omat[ii, ii] = self.ns[ii]

        self.h2omat[0, 2] = -self.h2omat[2, 2]
        self.h2omat[1, 3] = -3 * self.h2omat[3, 3]
        self.h2omat[2, 4] = -6 * self.h2omat[4, 4]
        self.h2omat[0, 4] = 3 * self.h2omat[4, 4]
        self.h2omat[3, 5] = -10 * self.h2omat[5, 5]
        self.h2omat[1, 5] = 15 * self.h2omat[5, 5]
        self.h2omat[4, 6] = -15 * self.h2omat[6, 6]
        self.h2omat[2, 6] = 45 * self.h2omat[6, 6]
        self.h2omat[0, 6] = -15 * self.h2omat[6, 6]
        self.h2omat[5, 7] = -21 * self.h2omat[7, 7]
        self.h2omat[3, 7] = 105 * self.h2omat[7, 7]
        self.h2omat[1, 7] = -105 * self.h2omat[7, 7]
        self.h2omat[6, 8] = -28 * self.h2omat[8, 8]
        self.h2omat[4, 8] = 210 * self.h2omat[8, 8]
        self.h2omat[2, 8] = -420 * self.h2omat[8, 8]
        self.h2omat[0, 8] = 105 * self.h2omat[8, 8]
        self.h2omat = self.h2omat[0:self.maxdeg, 0:self.maxdeg]

    def settransmat(self):
        self.transmat = np.full((self.dof, self.dof), 1.)

        for row_index in self.index_map:
            for col_index in self.index_map:
                for d in range(self.dim):
                    self.transmat[self.index_map[row_index], self.index_map[col_index]] *= self.h2omat[row_index[d], col_index[d]]


    # evaluate the Hermite basis functions at points given in np.array x
    # x.shape[0] should be number of points
    # x.shape[1] should be number of dimensions, i.e., the d in R^d
    # the function outputs a matrix of dimension (numpts * self.dof)
    # multiplying this output matrix by a coefficient vector beta will then yield function evaluations at x
    def basis(self, xin):
        if isinstance(xin, np.ndarray):
            x = xin
        else:
            x = np.array(xin)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        hermout = np.full((x.shape[0], self.dof), 1.)

        Hcached = np.zeros((x.shape[0], self.dim, self.maxdeg))
        for d in range(self.maxdeg):
            Hcached[:, :, d] = hermite.hermite(x, d)

        for index in self.index_map:
            for d in range(self.dim):
                hermout[:, self.index_map[index]] *= Hcached[:, d, index[d]]

        return hermout

    # evaluate the gradient of the Hermite basis functions at points given in np.array x
    # x.shape[0] should be number of points
    # x.shape[1] should be number of dimensions, i.e., the d in R^d
    # the function outputs a matrix of dimension (self.dim * numpts * self.dof)
    # specifically, output[i] is the derivative with respect to x_i at x
    # multiplying this output matrix by a coefficient vector beta yields gradient at x
    def gradient(self, xin):
        if isinstance(xin, np.ndarray):
            x = xin
        else:
            x = np.array(xin)

        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        hermout = np.full((self.dim, x.shape[0], self.dof), 1.)

        # Hcached stores all the one-dimensional Hermite function evaluations we need
        # Hcached[i, j, k] = self.H(x[i, j], k)
        Hcached = np.zeros((x.shape[0], self.dim, self.maxdeg))
        for d in range(self.maxdeg):
            Hcached[:, :, d] = hermite.hermite(x, d)

        for derivdim in range(self.dim):
            for index in self.index_map:
                for d in range(self.dim):
                    if d==derivdim:
                        nf = self.ns[index[d]]/self.ns[index[d]-1]
                        hermout[derivdim, :, self.index_map[index]] *= nf*index[d]*Hcached[:, d, index[d]-1]
                    else:
                        hermout[derivdim, :, self.index_map[index]] *= Hcached[:, d, index[d]]

        return hermout



class Bridge:

    def __init__(self, bips, mcps, numsubs, ncores=1,
                 method="naive", drift=None, gvec=None, wantpaths=True):
        self.burninpaths = bips
        self.mcmcpaths = mcps
        self.numsubintervals = numsubs
        self.ncores = ncores
        self.method = method
        self.drift = drift
        self.gvec = gvec
        self.wp = wantpaths
        self.approx = None

    # creates brownian bridge interpolations for given start and end
    # time point t and value x.
    def brownianbridge(self, ind):
        h = self.tdiff[ind]
        tvec = self.tin[ind] + (1 + np.arange(self.numsubintervals)) * h

        # W ~ N(0, sqrt(h)*g)
        wincs = np.random.multivariate_normal(mean=np.zeros(self.dim),
                                              cov=h * np.diag(np.square(self.gvec)),
                                              size=self.numsubintervals)
        w = np.cumsum(wincs, axis=0).T

        bridge = self.xin[ind, :, None] + w
        bridge -= ((tvec - self.tin[ind]) / (self.tin[ind+1] - self.tin[ind])) * (w[:, self.numsubintervals - 1, None] + self.xin[ind, :, None] - self.xin[ind+1, :, None])

        # concatenate the starting point to the bridge
        tvec = np.concatenate((self.tin[[ind]], tvec)).T
        bridge = np.concatenate((self.xin[ind, :, None], bridge), axis=1).T
        return tvec, bridge

    # Girsanov likelihood is computed using
    def naivegirsanov(self, path):
        # path is of size (numsubintervals + 1, dim)
        # b is of size (numsubintervals + 1, dim)
        b = self.drift(path)
        u = np.dot(np.diag(np.power(self.gvec, -2)), b.T).T
        int1 = np.tensordot(u[:-1, :], np.diff(path, axis=0)) #, axes=((0,1),(0,1)))
        u2 = np.einsum('ij,ji->i', u.T, b)
        int2 = np.sum(0.5 * (u2[1:] + u2[:-1])) * (self.tdiff[0])
        r = int1 - 0.5 * int2
        return r

    def driftode(self, t, xin):
        if len(xin.shape) == 1:
            x = np.expand_dims(xin, axis=0)
        else:
            x = xin

        xdot = self.drift(x)
        if len(xin.shape) == 1:
            return xdot[0,:]
        else:
            return xdot

    # set up guided diffusion bridge
    # assumes drift etc has already been defined
    def guided(self, ind):
        h = self.tdiff[ind]
        tvec = self.tin[ind] + (np.arange(self.numsubintervals + 1)) * h

        # STEP 1, solve the ODE dx/dt = b(x) where b is the drift function
        # xtalt = scipy.integrate.solve_ivp(self.driftode,
        #                                   (self.tin[ind], self.tin[ind+1]),
        #                                   self.xin[ind],
        #                                   max_step=h,
        #                                   t_eval=tvec)
        # xt = xtalt.y
        xtalt = scipy.integrate.odeint(self.driftode, self.xin[ind], tvec, tfirst=True)
        xt = xtalt.T

        # STEP 2, evaluate two functions:
        # B(t) = V(x(t)) where V is the Jacobian of the drift function
        # beta(t) = b(x(t)) - B(t)*x(t)
        Br = self.grad(xt.T)
        B = np.transpose(Br, [1,2,0])
        Bxt = np.einsum('ijk,ki->ji', B, xt)
        beta = self.drift(xt.T).T - Bxt
        atilde = np.diag(self.gvec**2)

        # STEP 3, solve the Matrix ODE dK/dt = B(t)*K + K*B(t) - \tilde{a}
        Kt = np.zeros((self.numsubintervals + 1, self.dim, self.dim))
        Kt[self.numsubintervals] = np.zeros((self.dim, self.dim))

        # take an Euler step
        Kt[self.numsubintervals - 1] = h * atilde

        def Kvf(j):
            return (-1) * (np.dot(B[self.numsubintervals - j, :], Kt[self.numsubintervals - j, :]) + np.dot(Kt[self.numsubintervals - j, :], B[self.numsubintervals - j, :]) - atilde)

        # take an AB2 step
        part1 = (3 / 2) * Kvf(1)
        part2 = -(1 / 2) * Kvf(0)
        Kt[self.numsubintervals - 2] = Kt[self.numsubintervals - 1] + h * (part1 + part2)

        # take an AB3 step
        part1 = (23 / 12) * Kvf(2)
        part2 = -(4 / 3) * Kvf(1)
        part3 = (5 / 12) * Kvf(0)
        Kt[self.numsubintervals - 3] = Kt[self.numsubintervals - 2] + h * (part1 + part2 + part3)

        # go backwards to t=0
        # fourth-order Adams-Bashforth
        for j in range(self.numsubintervals - 3):
            part1 = (55 / 24) * Kvf(j + 3)
            part2 = -(59 / 24) * Kvf(j + 2)
            part3 = (37 / 24) * Kvf(j + 1)
            part4 = -(3 / 8) * Kvf(j)
            step = h * (part1 + part2 + part3 + part4)
            Kt[(self.numsubintervals - 4) - j] = Kt[(self.numsubintervals - 3) - j] + step

        # STEP 4, solve the ODE dv/dt = B(t)*v + \beta(t)
        vt = np.zeros((self.dim, self.numsubintervals + 1))
        vt[:, self.numsubintervals] = self.xin[ind+1]

        def vvf(j):
            return (-1) * (np.dot(B[self.numsubintervals - j, :], vt[:, self.numsubintervals - j]) + beta[:, self.numsubintervals - j])

        # take an Euler step
        vt[:, self.numsubintervals - 1] = vt[:, self.numsubintervals] + h * vvf(0)

        # take an AB2 step
        part1 = (3 / 2) * vvf(1)
        part2 = -(1 / 2) * vvf(0)
        vt[:, self.numsubintervals - 2] = vt[:, self.numsubintervals - 1] + h * (part1 + part2)

        # take an AB3 step
        part1 = (23 / 12) * vvf(2)
        part2 = -(4 / 3) * vvf(1)
        part3 = (5 / 12) * vvf(0)
        vt[:, self.numsubintervals - 3] = vt[:, self.numsubintervals - 2] + h * (part1 + part2 + part3)

        # go backwards to t=0
        # fourth-order Adams-Bashforth
        for j in range(self.numsubintervals - 3):
            part1 = (55 / 24) * vvf(j + 3)
            part2 = -(59 / 24) * vvf(j + 2)
            part3 = (37 / 24) * vvf(j + 1)
            part4 = -(3 / 8) * vvf(j)
            step = h * (part1 + part2 + part3 + part4)
            vt[:, (self.numsubintervals - 4) - j] = vt[:, (self.numsubintervals - 3) - j] + step

        # STEP 5, compute
        # r(s,x) = [K(s)]^{-1}(v(s) - x)
        # b^0(t,x) = b(x) + a*r(t,x)
        Ktinv = np.zeros((self.numsubintervals, self.dim, self.dim))
        for j in range(self.numsubintervals):
            try:
                Ktinv[j] = np.linalg.inv(Kt[j])  
            except np.linalg.LinAlgError:
                print("using pseudoinverse")
                Ktinv[j] = np.linalg.pinv(Kt[j]) # slight modification

        @lru_cache(maxsize=None)
        def r(j):
            if (j >= 0) or (j < self.numsubintervals):
                return np.dot(Ktinv[j], (vt[:, j] - self.traj[:,j]))
            else:
                print("Index out of range")

        @lru_cache(maxsize=None)
        def bcirc(j):
            return self.drift(self.traj[:,[j]].T) + (self.gvec**2) * r(j)

        def proposal(start, end):
            doneFlag = 0
            while (doneFlag == 0):
                doneFlag = 1
                self.traj = np.zeros((self.dim, self.numsubintervals + 1))
                self.traj[:, 0] = start
                self.traj[:, self.numsubintervals] = end
                h12 = np.sqrt(h)
                for j in range(self.numsubintervals - 1):
                    self.traj[:, j + 1] = self.traj[:, j] + h * bcirc(j) + h12 * self.gvec * np.random.randn(self.dim)
                    if np.linalg.norm(self.traj[:, j + 1]) > 10:
                        doneFlag = 0
                        break

        # compute MCMC accept/reject likelihood
        @lru_cache(maxsize=None)
        def vode(j):
            return np.dot(B[j], self.traj[:,j]) + beta[:, j]

        def loglik():
            # compute G and int0T using Simpson's rule
            int0T = 0
            for i in range(self.numsubintervals):
                if (i == 0) or (i == (self.numsubintervals - 1)):
                    w = 1 / 2
                elif (i % 2) == 0:
                    w = 2 / 3
                else:
                    w = 4 / 3

                term1 = (self.drift(self.traj[:, [i]].T) - vode(i))
                term2 = r(i)
                int0T += h * w * np.dot(term1, term2)

            return np.asscalar(int0T)

        if self.wp:
            samples = np.zeros((self.mcmcpaths, self.numsubintervals+1, self.dim))
        else:
            mmat = np.zeros((self.approx.dof, self.approx.dof))
            rvec = np.zeros((self.approx.dof, self.approx.dim))

        proposal(self.xin[ind], self.xin[ind+1])
        oldtraj = self.traj.copy()
        oldlik = loglik()
        arburn = np.zeros(self.burninpaths, dtype='int32')
        arsamp = np.zeros(self.mcmcpaths, dtype='int32')
        for j in range(self.burninpaths + self.mcmcpaths):
            proposal(self.xin[ind], self.xin[ind+1])
            proptraj = self.traj.copy()
            thislik = loglik()
            rho = thislik - oldlik
            if rho > np.log(np.random.uniform(size=1)[0]):
                oldlik = thislik
                if j < self.burninpaths:
                    arburn[j] = 1
                else:
                    arsamp[j - self.burninpaths] = 1
                curtraj = proptraj
            else:
                curtraj = oldtraj

            if j >= self.burninpaths:
                # either save this path or save statistics from path
                if self.wp:
                    samples[j - self.burninpaths, :, :] = curtraj.T
                else:
                    pp = self.approx.basis(curtraj.T[:-1])
                    pp2 = pp.copy()
                    for ii in range(pp.shape[1]):
                        pp2[:, ii] *= np.diff(tvec)

                    mmat += np.matmul(pp.T, pp2) / self.mcmcpaths
                    rvec += np.matmul(pp.T, np.diff(curtraj.T, axis=0)) / self.mcmcpaths

        if self.wp:
            return samples, np.mean(arburn), np.mean(arsamp)
        else:
            return mmat, rvec, np.mean(arburn), np.mean(arsamp)

    # make one naive diffusion bridge
    def naive(self, ind):
        if self.wp:
            samples = np.zeros((self.mcmcpaths, self.numsubintervals+1, self.dim))
        else:
            mmat = np.zeros((self.approx.dof, self.approx.dof))
            rvec = np.zeros((self.approx.dof, self.approx.dim))

        _, xcur = self.brownianbridge(ind)
        oldlik = self.naivegirsanov(xcur)

        arburn = np.zeros(self.burninpaths)
        for jj in range(self.burninpaths):
            btvec, prop = self.brownianbridge(ind)
            proplik = self.naivegirsanov(prop)

            rho = proplik - oldlik
            if (rho > np.log(np.random.uniform())):
                xcur = prop
                oldlik = proplik
                arburn[jj] = 1

        # for each path being sampled (r = 0 to r = R)
        arsamp = np.zeros(self.mcmcpaths)
        for jj in range(self.mcmcpaths):
            btvec, prop = self.brownianbridge(ind)
            proplik = self.naivegirsanov(prop)

            rho = proplik - oldlik
            if (rho > np.log(np.random.uniform())):
                xcur = prop
                oldlik = proplik
                arsamp[jj] = 1

            if self.wp:
                samples[jj,:,:] = xcur
            else:
                pp = self.approx.basis(xcur[:-1])
                pp2 = pp.copy()
                for ii in range(pp.shape[1]):
                    pp2[:,ii] *= np.diff(btvec)

                mmat += np.matmul(pp.T, pp2) / self.mcmcpaths
                rvec += np.matmul(pp.T, np.diff(xcur, axis=0)) / self.mcmcpaths

        if self.wp:
            return samples, np.mean(arburn), np.mean(arsamp)
        else:
            return mmat, rvec, np.mean(arburn), np.mean(arsamp)

    # assume that allx is numpts by dim
    # assume that allt is numpts by 1
    def diffbridge(self, allx, allt):
        # save the data
        self.xin = allx
        self.tin = allt

        # set numpts, i.e., number of points in time series
        self.numpts = allx.shape[0]

        # set dimension based on data
        self.dim = allx.shape[1]

        # one time series, one interval, all dimensions at a time
        # this vector stores the time step *sizes* that the bridge should take,
        # in between each pair of prescribed points
        self.tdiff = np.diff(allt) / self.numsubintervals

        # create pool and assign tasks
        # one index per path
        allinds = np.arange((self.numpts-1), dtype='int32')
        p = multiprocessing.Pool(self.ncores)
        if (self.method == "naive"):
            allouts = p.map(self.naive, allinds)
            # allouts = map(self.naive, allinds)
            p.close()
            p.join()
        elif (self.method == "guided"):
            allouts = p.map(self.guided, allinds)
            # allouts = map(self.guided, allinds)
            p.close()
            p.join()

        # process everything we got from the map
        if (self.wp):
            allbridges, allburnaccepts, allsampaccepts = zip(*allouts)
        else:
            allmmat, allrvec, allburnaccepts, allsampaccepts = zip(*allouts)
            allmmat = np.sum(np.stack(allmmat),axis=0)
            allrvec = np.sum(np.stack(allrvec),axis=0)
            allbridges = (allmmat, allrvec)

        self.meanBurn = np.mean(np.array(allburnaccepts))
        self.meanSamp = np.mean(np.array(allsampaccepts))

        return allbridges



if __name__ == "__main__":

    npts = 192
    ndim = 2
    dt = 0.001
    gnumsteps = 4800
    numreps = 10
    saveint = gnumsteps//npts
    t = np.arange(npts+1, dtype='int64')*dt*saveint
    
    hermdrift = Hermite(3,2)
    hbeta = np.random.randn(hermdrift.dof, ndim)

    emmaxiter = 10

    for jjj in range(emmaxiter):
        def hdrift(x):
            return np.matmul(hermdrift.basis(x), hbeta)

        def gradhdrift(x):
            return np.matmul(hermdrift.gradient(x), hbeta)

        myherm = Hermite(3,2)
        mybridge = Bridge(20,80,100,method="guided",ncores=12,wantpaths=False)
        mybridge.drift = hdrift
        mybridge.grad = gradhdrift
        mybridge.gvec = np.array([0.75, 0.75])
        mybridge.approx = myherm

        mmat = np.zeros((myherm.dof, myherm.dof))
        rvec = np.zeros((myherm.dof, myherm.dim))

        for iii in range(numreps):
            print(jjj, iii)
            fname = "./npy.a3/data" + str(iii).rjust(3,"0") + ".npy"
            x = np.load(fname)
            samples = mybridge.diffbridge(x, t)
            print(mybridge.meanBurn)
            print(mybridge.meanSamp)
            mmat += samples[0]
            rvec += samples[1]

        fname = "mmat" + str(jjj).rjust(3,"0")
        np.save(fname, samples[0])
        fname = "rvec" + str(jjj).rjust(3,"0")
        np.save(fname, samples[1])
        hbetanew = np.linalg.solve(mmat, rvec)
        fname = "hbetanew" + str(jjj).rjust(3,"0")
        np.save(fname, hbetanew)
        change = np.linalg.norm(hbeta - hbetanew)
        print(hbetanew, change)
        hbeta = hbetanew
        if change < 5.0e-4:
            break



""""
    # print(samples[:,0,:])
    # print(samples[:,400,:])

    print(mybridge.meanBurn)
    print(mybridge.meanSamp)

    for j in range(mybridge.numpts-1):
        for i in range(mybridge.mcmcpaths):
            plt.plot(samples[j,i,:,0], samples[j,i,:,1])

    plt.show()
"""


# PART 4:
# create a new class to generate data for all our different models


