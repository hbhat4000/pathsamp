import numpy as np
import cProfile
import time
import sys

def H(degree, x):
    switcher = {
        0: 0.63161877774606470129,
        1: 0.63161877774606470129 * x,
        2: 0.44662192086900116570 * (np.power(x, 2) - 1),
        3: 0.25785728623970555997 * (np.power(x, 3) - 3 * x),
        4: 0.12892864311985277998 * (np.power(x, 4) - 6 * np.power(x, 2) + 3),
    }
    return switcher.get(degree, "Polynomial degree exceeded")

def hermite_basis3(x):
    y = np.zeros((x.shape[0], 20))
    index = 0

    for d in range(0, 4):
        for k in range(0, d + 1):
            for j in range(0, d + 1):
                for i in range(0, d + 1):
                    if (i + j + k == d):
                        # print("d", d, "i", i, "j", j, "k", k, "index", index)
                        y[:, index] = H(i, x[:, 0]) * H(j, x[:, 1]) * H(k, x[:, 2])
                        index += 1

    return y

# this is the FAST one
def hermite_basis6(x):
    y = np.zeros((x.shape[0], 84))
    ynew = np.zeros((x.shape[0], 84))
    index = 0

    Hten = np.zeros((4, x.shape[0], x.shape[1]))
    for d in range(0, 4):
        Hten[d,:,:] = H(d, x)

    for d in range(0, 4):
        for n in range(0, d + 1):
            for m in range(0, d + 1):
                for l in range(0, d + 1):
                    for k in range(0, d + 1):
                        for j in range(0, d + 1):
                            for i in range(0, d + 1):
                                if (i + j + k + l + m + n == d):
                                    # print("d", d, "i", i, "j", j, "k", k, "l", l, "m", m, "n", n, "index", index)
                                    # y[:, index] = H(i, x[:, 0]) * H(j, x[:, 1]) * H(k, x[:, 2]) * H(l, x[:, 3]) * H(m, x[:, 4]) * H(n, x[:, 5])
                                    y[:,index]=Hten[i,:,0]*Hten[j,:,1]*Hten[k,:,2]*Hten[l,:,3]*Hten[m,:,4]*Hten[n,:,5]
                                    # print(np.sum(np.abs(y - ynew)))
                                    index += 1

    return y

# this is the SLOW one
def hermite_basis6new(x):
    y = np.zeros((x.shape[0], 84))
    index = 0

    for d in range(0, 4):
        for n in range(0, d + 1):
            mub = d + 1 - n
            for m in range(0, mub):
                lub = mub - m
                for l in range(0, lub):
                    kub = lub - l
                    for k in range(0, kub):
                        jub = kub - k
                        for j in range(0, jub):
                            iub = jub - j
                            for i in range(0, iub):
                                tally = i + (d + 1) - iub
                                if (tally == d):
                                    # print("d", d, "i", i, "j", j, "k", k, "l", l, "m", m, "n", n, "index", index)
                                    y[:, index] = H(i, x[:, 0]) * H(j, x[:, 1]) * H(k, x[:, 2]) * H(l, x[:, 3]) * H(m, x[:, 4]) * H(n, x[:, 5])
                                    index += 1

    return y

def test3(n):
    t0 = time.time()
    x3 = np.random.randn(n, 3)
    t1 = time.time()
    print("np.random.randn(n,3): ", t1-t0)

    t0 = time.time()
    y3 = hermite_basis3(x3)
    t1 = time.time()
    print("hermite_basis3(x3): ", t1-t0)

def test6(n):
    t0 = time.time()
    x6 = np.random.randn(n, 6)
    t1 = time.time()
    print("np.random.randn(n,6): ", t1-t0)

    t0 = time.time()
    y6 = hermite_basis6(x6)
    t1 = time.time()
    print("hermite_basis6(x3): ", t1-t0)

def test6new(n):
    t0 = time.time()
    x6 = np.random.randn(n, 6)
    t1 = time.time()
    print("np.random.randn(n,6): ", t1-t0)

    t0 = time.time()
    y6 = hermite_basis6new(x6)
    t1 = time.time()
    print("hermite_basis6new(x3): ", t1-t0)

def diff6(n):
    x6 = np.random.randn(n, 6)
    y6 = hermite_basis6(x6)
    y6new = hermite_basis6new(x6)
    diff = np.sum(np.abs(y6 - y6new))
    print("Error: ", diff)

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        n = int(sys.argv[1])
    else:
        n = 10

    if len(sys.argv) >= 3:
        whichone = int(sys.argv[2])
    else:
        whichone = 3

    if whichone==6:
        print("test6")
        test6(n)
    elif whichone==7:
        print("test6new")
        test6new(n)
    elif whichone==8:
        print("diff6")
        diff6(n)
    else:
        print("test3")
        test3(n)


