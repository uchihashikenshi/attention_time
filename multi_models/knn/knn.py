# coding:utf-8
import numpy
import fastdtw


def get_nearest_n_dtw(self, train, label, test):
    """
    :param train: Training dataset. Must be pandas object.
    :param label: Training label.
    :param test: Some point of test data. Must be numpy array object.
    :return: The nearest points of training dataset (with DTW metrics).
    """
    nn_dist_array, nn_ts_ls, nn_label_array = numpy.array([]), [], numpy.array([])
    te_ele = numpy.array(test).reshape(-1, 1)

    for i, tr_ele in enumerate(train):

        # sys.stdout.write('\r%d' % i)
        # sys.stdout.flush()

        tr_ele_ls = tr_ele.tolist()
        tr_ele = numpy.array(tr_ele).reshape(-1, 1)
        dist, path = fastdtw(te_ele, tr_ele, dist=euclidean)

        if len(nn_dist_array) < self.nn_num:
            nn_dist_array = numpy.append(nn_dist_array, dist)
            nn_ts_ls.append(tr_ele_ls)
            nn_label_array = numpy.append(nn_label_array, label[i])
        elif numpy.max(nn_dist_array) > dist:

            if numpy.max(nn_dist_array) < self.max_dist:
                break

            max_ind = numpy.argmax(nn_dist_array)
            nn_dist_array[max_ind] = dist
            nn_ts_ls[max_ind] = tr_ele_ls
            nn_label_array[max_ind] = label[i]
        else:
            continue
    nn_ts_array = numpy.array(nn_ts_ls)

    return nn_dist_array, nn_ts_array, nn_label_array