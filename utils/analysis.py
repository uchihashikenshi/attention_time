#coding:utf-8
import numpy
import six

class Analysis():

    def __init__(self):
        pass


    def find_pattern(self, ts_array, label_array, pred_proba, type='miss'):

        N_test = label_array.shape[0]
        pattern_x, pattern_y, pattern_ind = [], [], []

        for i in six.moves.range(N_test):
            pred = numpy.argmax(pred_proba[i])

            if type == 'miss':

                if pred == label_array[i]:
                    continue
                else:
                    pattern_x.append(ts_array[i])
                    pattern_y.append(label_array[i])
                    pattern_ind.append(i)
            else:

                if pred != label_array[i]:
                    continue
                else:
                    pattern_x.append(ts_array[i])
                    pattern_y.append(label_array[i])
                    pattern_ind.append(i)

        return pattern_x, pattern_y, pattern_ind


    def model_diff(self, model_pred1, model_pred2, label):

        diff, sum = 0, 0
        for pred1, pred2 in zip(model_pred1, model_pred2):
            sum += 1
            if pred1 == pred2:
                diff += 1
        diff_rate = diff / sum

        return diff_rate

