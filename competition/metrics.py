import numpy as np
import sklearn.metrics
import tabulate

"""
Python compute equal error rate (eer)
ONLY tested on binary classification

:param label: ground-truth label, should be a 1-d list or np.array, each element represents the ground-truth label of one sample
:param pred: model prediction, should be a 1-d list or np.array, each element represents the model prediction of one sample
:param positive_label: the class that is viewed as positive class when computing EER
:return: equal error rate (EER)
"""
def compute_eer(label, pred, positive_label=1):
    # all fpr, tpr, fnr, fnr, threshold are lists (in the format of np.array)
    # fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred)
    fpr, tpr, threshold = sklearn.metrics.roc_curve(label, pred, pos_label=positive_label)
    fnr = 1 - tpr

    # the threshold of fnr == fpr
    eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]

    # theoretically eer from fpr and eer from fnr should be identical but they can be slightly differ in reality
    eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    # return the mean of eer from fpr and from fnr
    eer = (eer_1 + eer_2) / 2
    # log fpr, fnr, eer by tabulate
    table = [["fpr", "fnr", "eer"], [eer_1, eer_2, eer]]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="fancy_grid"))

    return eer
    
def compute_acc(label, pred, threshold=0.5):
    pred = np.array(pred)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    acc = np.sum(pred == label) / len(label)
    table = [["threshold", "acc"], [threshold, acc]]
    print(tabulate.tabulate(table, headers="firstrow", tablefmt="fancy_grid"))
    return acc