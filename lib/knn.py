import numpy as np
import util

class KNN:

    def __init__(self, n_neighbors=5, distance_metric='euclidean',
                 distance_weighted=False):
        self.n_neighbors = n_neighbors
        self.distance_weighted = distance_weighted

        if distance_metric == 'euclidean':
            self.distance = util.euclidean_distance
        else:
            raise ValueError('invalid distance metric')

    def fit(self, examples, labels):
        self.neighbors = []
        for example, label in zip(examples, labels):
            self.neighbors.append(Neighbor(example, label))

    def k_neighbors(self, example):
        sorted_neighbors = sorted(
            self.neighbors,
            key=lambda neighbor: self.distance(neighbor.example, example)
        )

        return sorted_neighbors[:self.n_neighbors]

    def predict(self, example):
        label_counts = {}
        nearest_neighbors = self.k_neighbors(example)
        for neighbor in nearest_neighbors:
            weight = 1
            if self.distance_weighted:
                dist = self.distance(example, neighbor.example)
                if dist == 0.0:
                    return neighbor.label
                weight = 1.0 / dist

            if neighbor.label in label_counts:
                label_counts[neighbor.label] += weight
            else:
                label_counts[neighbor.label] = weight

        return sorted(label_counts, key=label_counts.__getitem__)[-1]

class Neighbor:

    def __init__(self, example, label):
        self.example = example
        self.label = label
