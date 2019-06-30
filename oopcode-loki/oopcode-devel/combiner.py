import numpy as np
import oopcode as oc

mmat = np.load("mmat000.npy")
rvec = np.load("rvec000.npy")
for iii in range(1,100):
    fname = "mmat" + str(iii).rjust(3,"0") + ".npy"
    mmat += np.load(fname)
    fname = "rvec" + str(iii).rjust(3,"0") + ".npy"
    rvec += np.load(fname)

beta = np.linalg.solve(mmat, rvec)
myherm = oc.Hermite(3,2)
print(np.dot(myherm.transmat, beta))



