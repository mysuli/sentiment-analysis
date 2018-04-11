# -*- coding: utf-8 -*-

import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def pred_evaluation(pred, label_test, roc=True, savepath='./'):
    '''
    Function to analysis predictive ability of a given classifier

    pred: list; predictions of test set
    label_test: list; true labels of test set
    roc: Bool; if true draw the roc curve and save the auc value
    savepath: str or None; path to place the output file

    return: dict; indice names as keys and indice values as dict values
    '''
    rtn = dict()
    rtn["Accuracy"] = np.mean(pred == label_test)
    rtn["Precision"] = metrics.precision_score(label_test, pred)  # P = TP/(TP+FP)
    rtn["Recall Rate"] = metrics.recall_score(label_test, pred)  # R = TP/(TP+FN)
    rtn["F1 Score"] = metrics.f1_score(label_test, pred)
    if roc:
        # draw the ROC curve
        fpr, tpr, thresholds = roc_curve(label_test, pred, pos_label=1)  # fpr:假阳性率 tpr:真阳性率
        rtn["AUC"] = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='area = {}'.format(rtn["AUC"]))
        plt.legend(loc="lower right")
        if savepath:
            plt.savefig('{}/rf_roc.png'.format(savepath))
    if savepath:
        with open('{}/rf_result.csv'.format(savepath), 'w+', encoding='utf-8') as f:
            for k, v in rtn.items():
                f.writelines("{}: {}\n".format(k, v))
    return rtn


