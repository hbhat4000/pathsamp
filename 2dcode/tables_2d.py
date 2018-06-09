import numpy as np
import pickle

test_names = ['./random_timepoints/rand_', './varying_subintervals/tp_11/subint_', './varying_num_timeseries/ts_', './varying_noise/noise_']

test_list = []
for test in test_names:
	meta_list = []
	for i in range(1, 11):
		if (test == './random_timepoints/rand_'):
			parvalue = i*10+1
		if (test == './varying_noise/noise_'):
			parvalue = i - 1
			if (parvalue > 7):
				break
		if (test == './varying_subintervals/tp_11/subint_' or test == './varying_num_timeseries/ts_'):
			parvalue = i

		with open(test + str(parvalue) + '.pkl', 'rb') as f:
			x, _, _, _, _, _, _, errors, em_param, _, _, sim_param = pickle.load(f)
		meta_list.append((errors, x.shape, em_param, sim_param))
	test_list.append(meta_list)

# 1) Hermite errors
print('\n ####################################################################################### \n')
print('1) Hermite errors')

# 1a) random timepoints
print('\n 1a) Error comparison for varying time points \n')
hermite_error_rand = np.zeros(10)
for j in range(10):
	hermite_error_rand[j] = test_list[0][j][0][1]
print(hermite_error_rand)

# 1b) num subintervals
print('\n 1b) Error comparison for varying number of subintervals')
hermite_error_sint = np.zeros(10)
for j in range(10):
	hermite_error_sint[j] = test_list[1][j][0][1]
print(hermite_error_sint)

# 1c) num time series
print('\n 1c) Error comparison for varying number of time series')
hermite_error_ts = np.zeros(10)
for j in range(10):
	hermite_error_ts[j] = test_list[2][j][0][1]
print(hermite_error_ts)

# 1d) noise
print('\n 1d) Error comparison for varying noise')
hermite_error_noise = np.zeros(8)
for j in range(8):
	hermite_error_noise[j] = test_list[3][j][0][1]
print(hermite_error_noise)

hermite_error_list = [hermite_error_rand, hermite_error_sint, hermite_error_ts, hermite_error_noise]
################################################################################################

# 2) Ordinary errors
print('\n ####################################################################################### \n')
print('2) Ordinary errors')

# 2a) random timepoints
print('\n 2a) Error comparison for varying time points \n')
ordinary_error_rand = np.zeros(10)
for j in range(10):
	ordinary_error_rand[j] = test_list[0][j][0][0]
print(ordinary_error_rand)

# 2b) num subintervals
print('\n 2b) Error comparison for varying number of subintervals')
ordinary_error_sint = np.zeros(10)
for j in range(10):
	ordinary_error_sint[j] = test_list[1][j][0][0]
print(ordinary_error_sint)

# 2c) num time series
print('\n 2c) Error comparison for varying number of time series')
ordinary_error_ts = np.zeros(10)
for j in range(10):
	ordinary_error_ts[j] = test_list[2][j][0][0]
print(ordinary_error_ts)

# 2d) noise
print('\n 2d) Error comparison for varying noise')
ordinary_error_noise = np.zeros(8)
for j in range(8):
	ordinary_error_noise[j] = test_list[3][j][0][0]
print(ordinary_error_noise)

ordinary_error_list = [ordinary_error_rand, ordinary_error_sint, ordinary_error_ts, ordinary_error_noise]

#######################################################################################################

# 3) Gvec errors
print('\n ####################################################################################### \n')
print('3) Gvec errors')

# 3a) random timepoints
print('\n 3a) Error comparison for varying time points \n')
g_error_rand = np.zeros(10)
for j in range(10):
	g_error_rand[j] = np.sqrt(np.sum(np.square(np.abs(test_list[0][j][0][4]))))
print(g_error_rand)

# 3b) num subintervals
print('\n 3b) Error comparison for varying number of subintervals')
g_error_sint = np.zeros(10)
for j in range(10):
	g_error_sint[j] = np.sqrt(np.sum(np.square(np.abs(test_list[1][j][0][4]))))
print(g_error_sint)

# 3c) num time series
print('\n 2c) Error comparison for varying number of time series')
g_error_ts = np.zeros(10)
for j in range(10):
	g_error_ts[j] = np.sqrt(np.sum(np.square(np.abs(test_list[2][j][0][4]))))
print(g_error_ts)

# 3d) noise
print('\n 2d) Error comparison for varying noise')
g_error_noise = np.zeros(8)
for j in range(8):
	g_error_noise[j] = np.sqrt(np.sum(np.square(np.abs(test_list[3][j][0][4]))))
print(g_error_noise)

g_error_list = [g_error_rand, g_error_sint, g_error_ts, g_error_noise]

#######################################################################################################

# save to pickle file for future reference!
with open('results_2d.pkl','wb') as f:
    pickle.dump([hermite_error_list, ordinary_error_list, g_error_list], f)
