# coding:utf-8
import numpy
import sys
from fastdtw import fastdtw


class KNearestNeighbor(object):

    def __init__(self):
        pass

    def param_new(self, k, output_dim):
        self.k = k
        self.output_dim = output_dim

    def fit(self, test, train, label, k, output_dim, metrics="euclidean"):
        """
        :param test: Test data. Must be Numpy object.
        :param train: Labeled data. Must be Numpy object.
        :param label: Label for labeled data.
        :param k: Number of nearest neighbors.
        :param output_dim: Dimension of label.
        :param metrics: Distance measure of kNN. Default value is euclidean.
        :return: The nearest points of training data (with any metrics).
        """
        self.param_new(k, output_dim)
        nn_dist_array, nn_ts_array, nn_label_array = \
            numpy.array([]), numpy.empty((0, 30), int), numpy.empty((0, self.output_dim), int)

        for i, te_ele in enumerate(test):

            sys.stdout.write('\r%d' % i)
            sys.stdout.flush()

            for tr_ele, label_ele in zip(train, label):

                if metrics == "dtw":
                    dist, path = fastdtw(te_ele, tr_ele)
                elif metrics == "euclidean":
                    dist = self.euclidean(te_ele)
                else:
                    NotImplementedError()

                if len(nn_dist_array) < self.k:

                    nn_dist_array = numpy.append(nn_dist_array, dist)
                    nn_ts_array = numpy.append(nn_ts_array, numpy.array([tr_ele]), axis=0)
                    nn_label_array = numpy.append(nn_label_array, numpy.array([label_ele]), axis=0)
                elif numpy.max(nn_dist_array) > dist:

                    max_ind = numpy.argmax(nn_dist_array)
                    nn_dist_array[max_ind] = dist
                    nn_ts_array[max_ind] = tr_ele
                    nn_label_array[max_ind] = label_ele
                else:
                    continue

        return nn_dist_array, nn_ts_array, nn_label_array

    def euclidean(self, data):
        return 0

    def prediction(self):
        pass
