import sys
sys.path.append('../lib')

import numpy as np
import knn
import pca

from sklearn import cross_validation
from sklearn import preprocessing

# http://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
fname = 'data/semeion.csv'

n_instances = 1593

n_attributes = 256
attributes_cols = range(256)

n_labels = 10
label_cols = range(256, 256 + 10)

examples = np.genfromtxt(fname, delimiter=',', usecols=attributes_cols)
examples = preprocessing.MinMaxScaler().fit_transform(examples)
examples = pca.reduce_dim(examples, explained_var_ratio=0.9, debug=True)

labels_mat = np.genfromtxt(fname, dtype=None, delimiter=',', usecols=label_cols)
labels = np.array([np.nonzero(row == 1)[0][0] for row in labels_mat])

knn_classifier = knn.KNN(n_neighbors=3, distance_weighted=True)

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
