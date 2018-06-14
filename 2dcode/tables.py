import numpy as np
import pickle

def error_tables(exp, hermite_errors, ordinary_errors, estimated_theta, true_theta, parval, mapping, threshold):
    errors = [hermite_errors, ordinary_errors]
    titles = ['L1 norm', 'L2 norm', 'Precision', 'Recall', 'Accuracy', 'F1 score']
    space = ['Hermite', 'Ordinary']

    for fn in range(len(titles)):
        print('Space:', space[0], ', error:', titles[fn], ', threshold:', threshold[3], ', errors:', errors[0][3][fn][-1])

    print('true theta: ', true_theta)
    print('estimated theta')
    for val in range(len(mapping)):
        print('val: ', mapping[val], '\n', estimated_theta[val])

