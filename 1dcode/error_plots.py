import numpy as np
import pickle
from matplotlib import pyplot as plt

def error_plots(exp, hermite_errors, ordinary_errors, parval, mapping, threshold):
    errors = [hermite_errors, ordinary_errors]
    points = np.arange(0, parval, 1)
    titles = ['L1 norm error', 'L2 norm error', 'Precision', 'Recall', 'Accuracy', 'F1 score']
    file_name = ['_error_L1', '_error_L2', '_precision', '_recall', '_accuracy', '_F1']
    space = ['Hermite', 'Ordinary']

    for sp in range(len(space)):
        for fn in range(len(file_name)):
            fig = plt.figure()
            ax = fig.gca()
            for th in range(threshold.shape[0]):
                if (th == 0):
                    plt.plot(points, errors[sp][th][fn], 'bo--', label = 'No threshold')
                else:
                    plt.plot(points, errors[sp][th][fn], label='threshold ' + str(threshold[th]))
            plt.title(titles[fn] + ' in ' + space[sp] + ' space in 1D')
            plt.grid()
            ax.set_xticks(points)
            plt.xticks(points, mapping)
            plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
            plt.savefig('./' + exp + '/plots/' + space[sp] + file_name[fn]+ '.pdf', format = 'pdf', bbox_inches='tight')
            plt.close()
