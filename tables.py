import numpy as np
import pickle
from matplotlib import pyplot as plt

result_suffix = ['1d', '2d', '3d', '4d']
result_prefix = './tables/results_'

error_list = []
for res in result_suffix:
	with open(result_prefix + res + '.pkl', 'rb') as f:
		hermite, ordinary, g = pickle.load(f)
	error_list.append((hermite, ordinary, g))

herm_rand = np.zeros((10, len(result_suffix)))
herm_sint = np.zeros((10, len(result_suffix)))
herm_ts = np.zeros((10, len(result_suffix)))
herm_noise = np.zeros((8, len(result_suffix)))
i = 0
for err in error_list:
	herm_rand[:, i] = err[0][0]
	herm_sint[:, i] = err[0][1]
	herm_ts[:, i] = err[0][2]
	herm_noise[:, i] = err[0][3]
	i += 1

np.savetxt('./tables/herm_rand.csv', herm_rand)
np.savetxt('./tables/herm_sint.csv', herm_sint)
np.savetxt('./tables/herm_ts.csv', herm_ts)
np.savetxt('./tables/herm_noise.csv', herm_noise)

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(herm_rand[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for random time points in hermite space')
plt.grid()
plt.savefig('./tables/plots/herm_rand.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(herm_sint[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for varying intervals in hermite space')
plt.grid()
plt.savefig('./tables/plots/herm_sint.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(herm_ts[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for varying number of time series in hermite space')
plt.grid()
plt.savefig('./tables/plots/herm_ts.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(herm_noise[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for varying noise levels in hermite space')
plt.grid()
plt.savefig('./tables/plots/herm_noise.eps', format = 'eps', bbox_inches='tight')


ord_rand = np.zeros((10, len(result_suffix)))
ord_sint = np.zeros((10, len(result_suffix)))
ord_ts = np.zeros((10, len(result_suffix)))
ord_noise = np.zeros((8, len(result_suffix)))
i = 0
for err in error_list:
	ord_rand[:, i] = err[1][0]
	ord_sint[:, i] = err[1][1]
	ord_ts[:, i] = err[1][2]
	ord_noise[:, i] = err[1][3]
	i += 1

np.savetxt('./tables/ord_rand.csv', ord_rand)
np.savetxt('./tables/ord_sint.csv', ord_sint)
np.savetxt('./tables/ord_ts.csv', ord_ts)
np.savetxt('./tables/ord_noise.csv', ord_noise)

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(ord_rand[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for random time points in ordinary space')
plt.grid()
plt.savefig('./tables/plots/ord_rand.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(ord_sint[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for varying intervals in ordinary space')
plt.grid()
plt.savefig('./tables/plots/ord_sint.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(ord_ts[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for varying number of time series in ordinary space')
plt.grid()
plt.savefig('./tables/plots/ord_ts.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(ord_noise[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error for varying noise levels in ordinary space')
plt.grid()
plt.savefig('./tables/plots/ord_noise.eps', format = 'eps', bbox_inches='tight')


g_rand = np.zeros((10, len(result_suffix)))
g_sint = np.zeros((10, len(result_suffix)))
g_ts = np.zeros((10, len(result_suffix)))
g_noise = np.zeros((8, len(result_suffix)))
i = 0
for err in error_list:
	g_rand[:, i] = err[2][0]
	g_sint[:, i] = err[2][1]
	g_ts[:, i] = err[2][2]
	g_noise[:, i] = err[2][3]
	i += 1

np.savetxt('./tables/g_rand.csv', g_rand)
np.savetxt('./tables/g_sint.csv', g_sint)
np.savetxt('./tables/g_ts.csv', g_ts)
np.savetxt('./tables/g_noise.csv', g_noise)

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(g_rand[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error in gvec for random time points')
plt.grid()
plt.savefig('./tables/plots/g_rand.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(g_sint[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error in gvec for varying intervals')
plt.grid()
plt.savefig('./tables/plots/g_sint.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(g_ts[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error in gvec for varying number of time series')
plt.grid()
plt.savefig('./tables/plots/g_ts.eps', format = 'eps', bbox_inches='tight')

fig = plt.figure()
ax = fig.gca()
for i in range(len(result_suffix)):
	plt.plot(g_noise[:, i], label = result_suffix[i])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
plt.title('Error in gvec for varying noise levels')
plt.grid()
plt.savefig('./tables/plots/g_noise.eps', format = 'eps', bbox_inches='tight')
