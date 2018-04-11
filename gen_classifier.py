# -*- coding: utf-8 -*-

import logging
import numpy as np
from sklearn import metrics
from sklearn import cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def gen_rf(X_train, label_train, para_grid):
    '''
    Function to train a randomforest model and find optimal paras by gridsearch

    X_train: matrix; Design matrix of train set
    label_train: list; label of train set
    para_grid: dict; para names as keys, para value settings to be searched as value 

    return: rf model; a rf model with optimal paras found by gridsearch
    '''
    # generate the randomforest model
    np.random.seed(100)
    rf = RandomForestClassifier(random_state=10)
    # find the optimastic parameters by gridsearch
    rf = GridSearchCV(rf, para_grid, n_jobs=-1).fit(X_train, label_train)
    logging.info("{} found by GridSearch".format(rf.best_params_))

    return rf

