#!/usr/bin/env python 3.6.4 (v3.6.4:d48ecebad5)
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 14:17:11 2019
Implement Logistic Regression and Linear SVM in dating-full.csv 2
@author: liang257
"""

import pandas as pd
import numpy as np
import sys
import warnings
warnings.filterwarnings("ignore")


'''
(i) Training the logistic regression classifier
'''
def lr(trainingSet, testSet):
    '''Initialization'''
    L2_lambda = 0.01
    step_size = 0.01
    max_iter = 500
    tol = 1e-6
    y_train = trainingSet['decision']
    y_test = testSet['decision']
    x_train = trainingSet.drop('decision', axis = 1)
    dim = x_train.shape
    x_train['new_col'] = 1
    x_test = testSet.drop('decision', axis = 1)
    x_test['new_col'] = 1
    weight = np.zeros(dim[1]+1)

    i = 0
    while i < max_iter:
        i += 1

        # Make predictions given current w
        y_hat = 1 / (1 + np.exp(-np.dot(x_train, weight).astype(float)))

        # Calculate gradient for each parameter
        delta = np.dot(np.transpose(y_hat - y_train), x_train) + L2_lambda * weight

        #Move parameters in direction of gradient
        weight_new = weight - step_size * delta

        #Check the stop criteria
        if np.linalg.norm(weight - weight_new) < tol:
            weight = weight_new
            break

        weight = weight_new

    #Make predictions on train and test set
    y_hat = (1 / (1 + np.exp(-np.dot(x_train, weight).astype(float))) > 0.5).astype(int)
    train_acc = round(np.sum(y_hat == y_train)/dim[0], 2)
    y_hat = (1 / (1 + np.exp(-np.dot(x_test, weight).astype(float))) > 0.5).astype(int)
    test_acc = round(np.sum(y_hat == y_test) / len(y_test), 2)

    print("Training Accuracy LR:", train_acc)
    print("Testing Accuracy LR:", test_acc)

    return([train_acc, test_acc, weight])

'''
(ii) Train a linear SVM classifier
'''
def svm(trainingSet, testSet):
    '''Initialization'''
    svm_lambda = 0.01
    step_size = 0.5
    max_iter = 500
    tol = 1e-6
    y_train = (trainingSet['decision']-0.5) * 2
    y_test = (testSet['decision'] - 0.5) * 2
    x_train = trainingSet.drop('decision', axis=1)
    dim = x_train.shape
    x_train['new_col'] = 1
    x_test = testSet.drop('decision', axis=1)
    x_test['new_col'] = 1
    weight = np.zeros(dim[1] + 1)

    i=0
    while i < max_iter:
        i += 1
        #Make predictions given current w
        y_hat = np.dot(x_train, weight)

        #Calculate gradient for each parameter
        delta_i = x_train.values * y_train.values.reshape((dim[0],1))
        mask = (y_train * y_hat < 1).astype(int)
        delta_i = delta_i * mask.values.reshape((dim[0],1))
        delta = svm_lambda * weight - 1/dim[0] * (np.sum(delta_i, axis=0))

        # Move parameters in direction of gradient
        weight_new = weight - step_size * delta

        # Check the stop criteria
        if np.linalg.norm(weight - weight_new) < tol:
            weight = weight_new
            break

        weight = weight_new

    # Make predictions on train and test set
    y_hat = ((np.dot(x_train, weight) > 0) - 0.5) * 2
    train_acc = round(np.sum(y_hat == y_train) / dim[0], 2)
    y_hat = ((np.dot(x_test, weight) > 0) - 0.5) * 2
    test_acc = round(np.sum(y_hat == y_test) / len(y_test), 2)

    print("Training Accuracy SVM:", train_acc)
    print("Testing Accuracy SVM:", test_acc)

    return([train_acc, test_acc, weight])


if __name__ == '__main__':
    '''read arguments from the command line'''
    trainingDataFilename = sys.argv[1]
    testDataFilename = sys.argv[2]
    modelIdx = sys.argv[3]

    '''import dataset'''
    trainingSet = pd.read_csv(trainingDataFilename)
    testSet = pd.read_csv(testDataFilename)

    if int(modelIdx) == 1:
        lr(trainingSet, testSet)
    elif int(modelIdx) == 2:
        svm(trainingSet, testSet)
    else:
        print("ModelIdx can only take 1 or 2!!")

