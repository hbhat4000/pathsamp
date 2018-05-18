import numpy as np
import pickle

result_suffix = ['1d', '2d', '3ddd', '3dl', '4d']
result_prefix = './tables/results_'

error_list = []
for res in result_suffix:
	with open(result_prefix + res + '.pkl', 'rb') as f:
		hermite, ordinary, g = pickle.load(f)
	error_list.append((hermite, ordinary, g))

rand = np.zeros((10, len(result_suffix)))
sint = np.zeros((10, len(result_suffix)))
ts = np.zeros((10, len(result_suffix)))
noise = np.zeros((8, len(result_suffix)))
i = 0
for err in error_list:
	rand[:, i] = err[0][0]
	sint[:, i] = err[0][1]
	ts[:, i] = err[0][2]
	noise[:, i] = err[0][3]
	i += 1

np.savetxt('./tables/herm_rand.csv', rand)
np.savetxt('./tables/herm_sint.csv', sint)
np.savetxt('./tables/herm_ts.csv', ts)
np.savetxt('./tables/herm_noise.csv', noise)

rand = np.zeros((10, len(result_suffix)))
sint = np.zeros((10, len(result_suffix)))
ts = np.zeros((10, len(result_suffix)))
noise = np.zeros((8, len(result_suffix)))
i = 0
for err in error_list:
	rand[:, i] = err[0][0]
	sint[:, i] = err[0][1]
	ts[:, i] = err[0][2]
	noise[:, i] = err[0][3]
	i += 1

np.savetxt('./tables/ord_rand.csv', rand)
np.savetxt('./tables/ord_sint.csv', sint)
np.savetxt('./tables/ord_ts.csv', ts)
np.savetxt('./tables/ord_noise.csv', noise)


rand = np.zeros((10, len(result_suffix)))
sint = np.zeros((10, len(result_suffix)))
ts = np.zeros((10, len(result_suffix)))
noise = np.zeros((8, len(result_suffix)))
i = 0
for err in error_list:
	rand[:, i] = err[0][0]
	sint[:, i] = err[0][1]
	ts[:, i] = err[0][2]
	noise[:, i] = err[0][3]
	i += 1

np.savetxt('./tables/g_rand.csv', rand)
np.savetxt('./tables/g_sint.csv', sint)
np.savetxt('./tables/g_ts.csv', ts)
np.savetxt('./tables/g_noise.csv', noise)
