#!/usr/bin/env python 3.6.4 (v3.6.4:d48ecebad5)
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 14:23:08 2019
Preprocessing in dating-full.csv 1
@author: liang257
"""

import pandas as pd
import numpy as np

'''import dataset'''
data = pd.read_csv('dating-full.csv', nrows=6500)

'''
(i) Repeat the preprocessing steps in Assg2
'''
#Problem (i)
quote_count = 0
quote_attri = ['race', 'race_o', 'field']
for col in quote_attri:
    quote_count += sum(data[col].str.count("'")/2)
    data[col] = data[col].str.replace("'","")

#Problem (ii)
casechange_count = len(data['field'])-sum(data['field'].str.islower()) #count the number of cells that will be changed
data['field'] = data['field'].str.lower() #convert all the values in the column field to lowercase

#Problem (iv)
pso_participant = ["attractive_important", "sincere_important", "intelligence_important", "funny_important", "ambition_important", "shared_interests_important"]
pso_partner = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests"]

total1 = data[pso_participant].sum(axis=1)
data[pso_participant] = data[pso_participant].div(total1, axis = 0)

total2 = data[pso_partner].sum(axis=1)
data[pso_partner] = data[pso_partner].div(total2, axis = 0)

data_nbc = data.copy()

'''
(ii) Apply one-hot encoding to the categorical attributes
'''
cat_attri = ["gender", "race", "race_o", "field"]
output_dic = {"gender": "female",
              "race": "Black/African American",
              "race_o": "Other",
              "field": "economics"
              }

attri_numval = {}

for attri in cat_attri:
    #find the unique values in each attribute and sort them alphabetically
    uniq_level = sorted(data[attri].unique())
    l = len(uniq_level)

    for index, level in enumerate(uniq_level[:-1]):
        one_hot = np.zeros(l-1)
        one_hot[index] = 1
        attri_numval[level] = one_hot
        new_col = "_".join([attri, level])
        data[new_col] = (data[attri] == level).astype(int)

    attri_numval[uniq_level[-1]] = np.zeros(l)

    print("Mapped vector for " + output_dic[attri] + " in column " + attri + ": ", attri_numval[output_dic[attri]])

#drop the categorical data
data = data.drop(cat_attri, axis=1)

'''
(iii) Take random samples
'''
data_test = data.sample(random_state = 25, frac = 0.2)
data_train = data.drop(data_test.index)


'''
Prepare the trainingSet for NBC in 3
'''
#convert the categorical values in columns gender, race, race o and field to numeric values start from 0
map_attri = ["gender", "race", "race_o", "field"]
for col in map_attri:
    uniq_list = np.sort(data_nbc[col].unique())
    cat = 0
    for word in uniq_list:
        data_nbc[col] = data_nbc[col].replace(word, cat)
        cat += 1


data_nbc_test = data_nbc.sample(random_state = 25, frac = 0.2)
data_nbc_train = data_nbc.drop(data_test.index)



'''
ave the train and test set
'''
data_train.to_csv('trainingSet.csv', index=False)
data_test.to_csv('testSet.csv', index=False)
data_nbc_train.to_csv('trainingSet_nbc.csv', index=False)
data_nbc_test.to_csv('testSet_nbc.csv', index=False)