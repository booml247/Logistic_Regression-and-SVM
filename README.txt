NAME: Siqi Liang


INSTRUCTIONS:
1. I also generate two datasets trainingSet_nbc.csv and testSet_nbc.csv in preprocess-assg3.py to do the cv in part 3.
2. I run like this:
python3 preprocess-assg3.py
python3 lr_svm.py trainingSet.csv testSet.csv 1
python3 lr_svm.py trainingSet.csv testSet.csv 2
python3 cv.py

Details:
1. preprocess-assg3.py reads first 6500 data from the dataset and preprocesses the data by striping the surrounding quotes in the values for columns race, race_o and field, converting all the values in the column field to lowercase if they are not already in lowercases, normalizing values in prefer- ence scores of participant and preference scores of partner columns and applying one-hot encoding to the categorical attributes gender, race, race_o and field. The the dataset were divided into training set and test set.
2. lr_svm.py defines lr() function which can be used to train a logistic regression model and svm() function which can be used to train a SVM model.
3. cv.py conducts incremental 10-fold cross validation to plot learning curves for different classifiers, with training sets of varying size but constant test set size.
