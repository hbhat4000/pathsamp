import numpy as np
import pickle
from matplotlib import pyplot as plt

def error_tables(exp, hermite_errors, ordinary_errors, estimated_theta, true_theta, parval, mapping, threshold):
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
                if (sp == 0 and th == 3):
                    print('Space: ', space[sp], ', error: ', file_name[fn], ', threshold: ', threshold[th], ', errors: ', errors[sp][th][fn][-1])
                    print('estimated theta', estimated_theta[sp][th][fn])
                    print('true theta', true_theta[sp][th][fn])
