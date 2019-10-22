#!/usr/bin/env python 3.6.4 (v3.6.4:d48ecebad5)
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 20:18:23 2019
Learning Curves and Performance Comparison in dating-full.csv 3
@author: liang257
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nbc import nbc, discretize
from lr_svm import lr,svm
from scipy.stats import wilcoxon

'''
Import the dataset
'''
trainingSet = pd.read_csv('trainingSet.csv')
trainingSet_nbc = pd.read_csv('trainingSet_nbc.csv')

'''
(i) Split the dataset
'''
#shuffle the training data
trainingSet = trainingSet.sample(random_state=18, frac=1)
trainingSet_nbc = trainingSet_nbc.sample(random_state=18, frac=1)
trainingSet_nbc["gaming"] = trainingSet_nbc["gaming"].mask(trainingSet_nbc["gaming"] > 10, 10)
trainingSet_nbc["reading"] = trainingSet_nbc["reading"].mask(trainingSet_nbc["reading"] > 10, 10)
trainingSet_nbc = discretize(trainingSet_nbc, bin_nums=5)

#split the training data into 10 disjoint sets
split_size = 10
num_line = trainingSet.shape[0]
arr = np.arange(num_line)  # get a seq and set len=numLine
list_all = arr.tolist()
each_size = num_line // split_size

S = {}
S_nbc = {}
for i in range(split_size):
    S[i] = trainingSet.iloc[(i*each_size):((i+1)*each_size), ]
    S_nbc[i] = trainingSet_nbc.iloc[(i * each_size):((i + 1) * each_size), ]

'''
(ii) Compare NBC, LR, SVM
'''
t_frac = [0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
test_acc_nbc = pd.DataFrame(index = range(10), columns=t_frac)
test_acc_lr = pd.DataFrame(index = range(10), columns=t_frac)
test_acc_svm = pd.DataFrame(index = range(10), columns=t_frac)

for frac in t_frac:
    for index in range(split_size):
        print('frac:',frac, 'index:', index)
        #i
        test_set = S[index]
        test_set_nbc = S_nbc[index]
        #ii
        S_C = pd.DataFrame(columns=list(test_set))
        S_C_nbc = pd.DataFrame(columns=list(test_set_nbc))
        for i in range(split_size):
            if i == index:
                continue
            else:
                S_C = S_C.append(S[i])
                S_C_nbc = S_C_nbc.append(S_nbc[i])
        #iii
        train_set = S_C.sample(random_state=32, frac=frac)
        train_set_nbc = S_C_nbc.sample(random_state=32, frac=frac)

        '''apply nbc'''
        p_0, p_1, cpd = nbc(train_set_nbc, 1, bin_nums=5)

        # make predictions
        pred_test = []
        test_n = test_set_nbc.shape[0]
        for i in range(test_n):
            pred_0 = p_0
            pred_1 = p_1

            for col in test_set_nbc.columns[:-1]:
                pred_0 *= cpd[0][col].loc[test_set_nbc[col].iloc[i]]
                pred_1 *= cpd[1][col].loc[test_set_nbc[col].iloc[i]]

            pred_test.append(int(pred_1 > pred_0))

        test_acc = round(sum(pred_test == test_set_nbc["decision"]) / test_n, 2)
        test_acc_nbc.loc[index, frac] = test_acc
        print('Test Accuracy NBC:', test_acc)

        '''apply lr'''
        _, test_acc, _ = lr(train_set, test_set)
        test_acc_lr.loc[index, frac] = test_acc

        '''apply svm'''
        _, test_acc, _ = svm(train_set, test_set)
        test_acc_svm.loc[index, frac] = test_acc

test_rec_nbc = pd.DataFrame(index=t_frac, columns=['mean', 'sterr'])
test_rec_lr = pd.DataFrame(index=t_frac, columns=['mean', 'sterr'])
test_rec_svm = pd.DataFrame(index=t_frac, columns=['mean', 'sterr'])

for frac in t_frac:
    #calculate the mean
    test_rec_nbc['mean'].loc[frac] = np.mean(test_acc_nbc[frac])
    test_rec_lr['mean'].loc[frac]  = np.mean(test_acc_lr[frac])
    test_rec_svm['mean'].loc[frac]  = np.mean(test_acc_svm[frac])

    #calculate the standard error
    test_rec_nbc['sterr'].loc[frac]  = np.std(test_acc_nbc[frac])/np.sqrt(split_size)
    test_rec_lr['sterr'].loc[frac]  = np.std(test_acc_lr[frac])/np.sqrt(split_size)
    test_rec_svm['sterr'].loc[frac]  = np.std(test_acc_svm[frac])/np.sqrt(split_size)



'''
(iii) Plot the learning curves
'''
plt.figure()
plt.errorbar(np.asarray(t_frac)*num_line*0.9,test_rec_nbc['mean'],yerr=test_rec_nbc['sterr'],color='b',elinewidth=2,capsize=4,label='NBC')
plt.errorbar(np.asarray(t_frac)*num_line*0.9,test_rec_lr['mean'],yerr=test_rec_lr['sterr'],color='g',elinewidth=2,capsize=4,label='Logistic Regression')
plt.errorbar(np.asarray(t_frac)*num_line*0.9,test_rec_svm['mean'],yerr=test_rec_svm['sterr'],color='r',elinewidth=2,capsize=4,label='SVM')
#fmt :   'o' ',' '.' 'x' '+' 'v' '^' '<' '>' 's' 'd' 'p'
plt.xlabel('size of the training set')
plt.ylabel('model accuracy on test set')
plt.legend(loc='upper right')

plt.savefig('learning_curves.png')


'''
(v) Conduct paired Wilcoxon test
'''
res_nbc = []
res_lr = []
res_svm = []
for frac in t_frac:
    res_nbc.extend(test_acc_nbc[frac].values.tolist())
    res_lr.extend(test_acc_lr[frac].values.tolist())
    res_svm.extend(test_acc_svm[frac].values.tolist())


statistic, p_value = wilcoxon(np.asarray(res_nbc) - np.asarray(res_lr))
print('p-value of paired Wilcoxon test between NBC and LR: ', p_value)

statistic, p_value = wilcoxon(np.asarray(res_nbc) - np.asarray(res_svm))
print('p-value of paired Wilcoxon test between NBC and SVM: ', p_value)

statistic, p_value = wilcoxon(np.asarray(res_lr) - np.asarray(res_svm))
print('p-value of paired Wilcoxon test between LR and SVM: ', p_value)