import sys
sys.path.append('../lib')

import numpy as np
import knn

from sklearn import cross_validation
from sklearn import preprocessing

# http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
fname = 'data/letter.csv'

n_instances = 2000

n_attributes = 16
attributes_cols = range(1, 17)

n_labels = 26
label_col = 0

examples = np.genfromtxt(fname, delimiter=',', usecols=attributes_cols)
examples = preprocessing.MinMaxScaler().fit_transform(examples)

labels = np.genfromtxt(fname, dtype=None, delimiter=',', usecols=label_col)

knn_classifier = knn.KNN(n_neighbors=1)

kfolds = cross_validation.KFold(n_instances, n_folds=5, shuffle=True)
for train_indices, test_indices in kfolds:
    train_examples = examples[train_indices]
    train_labels = labels[train_indices]

    knn_classifier.fit(train_examples, train_labels)

    test_examples = examples[test_indices]
    test_labels = labels[test_indices]

    count = 0.0
    for example, label in zip(test_examples, test_labels):
        if knn_classifier.predict(example) == label:
            count += 1
    print "accuracy:", count / len(test_labels)
