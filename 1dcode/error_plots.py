import numpy as np
import pickle
from matplotlib import pyplot as plt

def theta_sparsity(theta, threshold):
    relative_threshold = threshold * np.max(np.abs(theta))
    theta[np.abs(theta) < relative_threshold] = 0.
    return theta

def compute_error(estimated, true, errors_computed, threshold):
    errors = []
    estimated = theta_sparsity(estimated, threshold)
    true = theta_sparsity(true, threshold)

    # regression metric
    # L1 norm
    if (errors_computed[0]):
        errors.append(np.sum(np.abs(true - estimated)))

    # L2 norm
    if (errors_computed[1]):
        errors.append(np.sqrt(np.sum(np.power(true - estimated, 2))))

    # classification metric, P = value is zero, N = value is non-zero
    # true positive => true was zero and estimated was zero
    TP = np.sum(np.logical_and(true == 0., estimated == 0.))
    # true negative => true was non-zero and estimated was non-zero
    TN = np.sum(np.logical_and(true != 0., estimated != 0.))
    # false positive => true was non-zero and estimated was zero
    FP = np.sum(np.logical_and(true != 0., estimated == 0.))
    # false negative => true was zero and estimated was non-zero
    FN = np.sum(np.logical_and(true == 0., estimated != 0.))

    # precision = true positives / total estimated positives {TP / (TP + FP)}
    if (errors_computed[2]):
        errors.append(TP / (TP + FP))

    # recall = true positives / total true positives {TP / (TP + FN)}
    if (errors_computed[3]):
        errors.append(TP / (TP + FN))
        
    # accuracy = total true / total predictions {(TP + TN) / (TP + TN + FP + FN)}
    if (errors_computed[4]):
        errors.append((TP + TN) / (TP + TN + FP + FN))

    # F1 score = 2*precision*recall / (precision + recall)
    if (errors_computed[5]):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        errors.append(2 * precision * recall / (precision + recall))

    return errors

def error_plots(exp, meta_error_list, parval, mapping, threshold):
    points = np.arange(0, parval, 1)
    error_plot = np.zeros((12, parval, threshold.shape[0]))
    errors_computed = np.array([True, True, True, True, True, True])

    for th in range(threshold.shape[0]):
        for i in range(parval):
            ordinary_errors = compute_error(meta_error_list[i][3].ordinary, meta_error_list[i][4].ordinary, errors_computed, threshold[th])
            hermite_errors = compute_error(meta_error_list[i][3].hermite, meta_error_list[i][4].hermite, errors_computed, threshold[th])

            # hermite errors
            error_plot[0, i, th] = hermite_errors[0] # L1 norm
            error_plot[1, i, th] = hermite_errors[1] # L2 norm
            error_plot[2, i, th] = hermite_errors[2] # precision
            error_plot[3, i, th] = hermite_errors[3] # recall
            error_plot[4, i, th] = hermite_errors[4] # accuracy
            error_plot[5, i, th] = hermite_errors[5] # F1 score

            # ordinary errors
            error_plot[6, i, th] = ordinary_errors[0] # L1 norm
            error_plot[7, i, th] = ordinary_errors[1] # L2 norm
            error_plot[8, i, th] = ordinary_errors[2] # precision
            error_plot[9, i, th] = ordinary_errors[3] # recall
            error_plot[10, i, th] = ordinary_errors[4] # accuracy
            error_plot[11, i, th] = ordinary_errors[5] # F1 score

    # 1) L1 norm error in estimated theta in Hermite space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[0, :, th], label='threshold ' + str(threshold[th]))
    plt.title('L1 norm error in Hermite space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/error_L1_hermite.eps', format = 'eps', bbox_inches='tight')

    # 2) L2 norm error in estimated theta in Hermite space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[1, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('L2 norm error in Hermite space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/error_L2_hermite.eps', format = 'eps', bbox_inches='tight')

    # 3) Precision in estimated theta in Hermite space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[2, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('Precision (TP / (TP + FP)) in Hermite space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/precision_hermite.eps', format = 'eps', bbox_inches='tight')

    # 4) Recall in estimated theta in Hermite space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[3, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('Recall (TP / (TP + FN)) in Hermite space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/recall_hermite.eps', format = 'eps', bbox_inches='tight')

    # 5) Accuracy in estimated theta in Hermite space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[4, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('Classification accuracy in Hermite space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/accuracy_hermite.eps', format = 'eps', bbox_inches='tight')

    # 6) Accuracy in estimated theta in Hermite space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[5, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('F1 score in Hermite space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/F1_hermite.eps', format = 'eps', bbox_inches='tight')

    # 7) L1 norm error in estimated theta in Ordinary space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[6, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('L1 norm error in Ordinary space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/error_L1_ordinary.eps', format = 'eps', bbox_inches='tight')

    # 8) L2 norm error in estimated theta in Ordinary space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[7, :, th], label = 'threshold '+ str(threshold[th]))
    plt.title('L2 norm error in Ordinary space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./'+ exp + '/plots/error_L2_ordinary.eps', format = 'eps', bbox_inches='tight')

    # 9) Precision in estimated theta in Ordinary space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[8, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('Classification precision (TP / (TP + FP)) in Ordinary space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/precision_ordinary.eps', format = 'eps', bbox_inches='tight')

    # 10) Recall in estimated theta in Ordinary space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[9, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('Classification recall (TP / (TP + FN)) in Ordinary space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/recall_ordinary.eps', format = 'eps', bbox_inches='tight')

    # 11) Accuracy in estimated theta in Ordinary space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[10, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('Classification accuracy in Ordinary space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/accuracy_ordinary.eps', format = 'eps', bbox_inches='tight')

    # 12) Accuracy in estimated theta in Ordinary space
    fig = plt.figure()
    ax = fig.gca()
    for th in range(threshold.shape[0]):
        plt.plot(points, error_plot[11, :, th], label = 'threshold ' + str(threshold[th]))
    plt.title('F1 score in Ordinary space')
    plt.grid()
    ax.set_xticks(points)
    plt.xticks(points, mapping)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)
    plt.savefig('./' + exp + '/plots/F1_ordinary.eps', format = 'eps', bbox_inches='tight')