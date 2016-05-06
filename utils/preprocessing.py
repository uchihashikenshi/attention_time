#coding:utf-8
import numpy
import sklearn
from sklearn.linear_model import LogisticRegression
import pandas as pd
import random
import os
import sys


class Preprocessing():
    """
    :param data: Input data. Must be pandas object.
    :param k_folds: Number of folds of cross validation. Must be at least 2.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, path):
        """
        :param path: Data path.
        :return: data: Loaded data.
        """

        data = open(self.data_dir + path)

        return data

    def calc_bookmark_num(self, data):
        """
        :param data: Opened text object of data.
        :return: vol_ls: List of bookmark volume of each web pages.
        """

        vol_ls = []

        for i, line in enumerate(data):
            vol = int(line.split(" ")[0])
            if i != 0:
                vol_ls.append(vol)

        return vol_ls

    def timeseries(self, data, sum_min=100):
        """
        :param data: Opened text object of data.
        :param sum_min: Threshold of bookmark volume.
        :return category_ts: List object of timeseries data divided by category.
        :return category_sum: List object of boookmark volume of each pages devided by category. (not used)
        """

        list, category_ts, category_sum = [], [], []

        for i, line in enumerate(data):
            if i == 0:
                continue

            elif 0 < i < 500001:
                # if i % 1000 == 0:
                #     sys.stdout.write("\r%d web page ended" % i)
                #     sys.stdout.flush()

                list.append(line.split(" "))
                sum = int(list[i-1][0])

                if sum > sum_min:
                    timeseries = [0 for j in xrange(1600)]

                    for bookmark_id_day in list[i - 1][1:]:
                        day = int(bookmark_id_day.split(":")[1])
                        timeseries[day] += 1

                    category_ts.append(numpy.array(timeseries))
                    category_sum.append(sum)

            else: break

        return category_ts, category_sum

    def div_train_test(self, category_ts, train_test_ratio):
        """
        :param category_ts: List object of timeseries data divided by category.
        :param train_test_ratio: Ratio of train data. Must be [0, 1] continuous value.
        :return category_ts_dict: Dict of timeseries divided by categories. Key is train and test.
        """

        category_ts_dict = {}

        random_ind = range(len(category_ts))
        random.shuffle(random_ind)

        train_ind = random_ind[:int(len(category_ts) * train_test_ratio)]
        test_ind = random_ind[int(len(category_ts) * train_test_ratio):]

        train_ts = numpy.array(category_ts)[train_ind]
        test_ts = numpy.array(category_ts)[test_ind]

        category_ts_dict.update({'train': train_ts})
        category_ts_dict.update({'test': test_ts})

        return category_ts_dict

    def make_started_ts(self, category_ts_dict):

        category_started_ts_dict = {}
        for data_type in ['train', 'test']:

            category_started_ts = []
            for i, ts in enumerate(category_ts_dict[data_type]):

                # if i % 100 == 0:
                #     sys.stdout.write("\r%d" % i)
                #     sys.stdout.flush()

                start_flag = 0  # bookmarkが開始したかフラグ
                started_ts = []  # bookmark開始以降の時系列を取得
                for day_b in ts:

                    if day_b > 0:
                        start_flag = 1
                    # bookmarkが開始されていれば, それ以降の時系列を60日ずつ取得.
                    # 30 - 1, 30 - 3, 30 - 7, 30 - 30に分割し学習データとする.

                    if start_flag:
                        started_ts.append(day_b)

                category_started_ts.append(started_ts)

            category_started_ts_dict.update({data_type: category_started_ts})

        return category_started_ts_dict

    def make_supervised_data(self, category_started_ts_dict, output_dim, stride=5, input_dim=30):

        category_input_ts_dict, category_label_dict, category_target_ts_dict, category_input_sum_dict = {}, {}, {}, {}
        for data_type in ['train', 'test']:

            category_input_ts, category_label, category_target_ts, category_input_sum = [], [], [], []
            target_dim = 2 * input_dim

            label_dim = input_dim / output_dim
            if input_dim % output_dim != 0:
                print "label_dim=%s: error" % label_dim
            data_num = 0
            for started_ts in category_started_ts_dict[data_type]:

                # 部分時系列の切り出し
                for i in xrange(200):

                    # stackoverflow阻止
                    if len(started_ts) < target_dim + stride * i:
                        break
                    else:
                        target_ts = numpy.array(started_ts[stride * i:target_dim + stride * i])
                        # ある程度bookmark数がある部分時系列のみ採用
                        input_average = numpy.average(target_ts[:input_dim])
                        input_sum = numpy.sum(numpy.array(target_ts[:input_dim]))

                        if input_average > 2:

                            label_ls = []
                            input_ts = target_ts[:input_dim]
                            for dim in xrange(output_dim):
                                output_ts = target_ts[input_dim + label_dim * dim:input_dim + label_dim * (dim + 1)]
                                output_average = numpy.average(output_ts)

                                # 二値分類
                                if input_average < output_average:
                                    label = 1.0
                                else:
                                    label = 0.0
                                label_ls.append(label)

                            category_input_ts.append(input_ts)
                            category_label.append(label_ls)
                            category_target_ts.append(target_ts)
                            # inputの総bookmark数はいらないけどモデルを分ける要因である可能性があるので解析のため残す
                            category_input_sum.append(numpy.array([input_sum]))

                            data_num += 1
                            # sys.stdout.write("\r%d data created" % data_num)
                            # sys.stdout.flush()

            index_ls = range(len(category_input_ts))
            random.shuffle(index_ls)

            category_input_ts_dict.update({data_type: numpy.array(category_input_ts)[index_ls]})
            category_label_dict.update({data_type: numpy.array(category_label)[index_ls]})
            category_target_ts_dict.update({data_type: numpy.array(category_target_ts)[index_ls]})
            category_input_sum_dict.update({data_type: numpy.array(category_input_sum)[index_ls]})

        return category_input_ts_dict, category_label_dict, category_target_ts_dict, category_input_sum_dict

    def make_shuffle_k_folds_data(self, x, y, k_folds=5, rand=False):
        """
        分割時にシャッフルするための関数. 現在は使用していない.
        :param x:
        :param y:
        :param k_folds:
        :param rand:
        :return:
        """

        batch_num = len(x) / k_folds
        index_ls = range(len(x))

        if rand:
            sys.stdout.write("\rrandomized")
            sys.stdout.flush()
            random.shuffle(index_ls)

        x, y = numpy.array(x), numpy.array(y)
        train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []

        for k in xrange(k_folds):
            test_index = index_ls[k*batch_num:(k+1)*batch_num]
            # valid_index = test_valid_index[:int(round(batch_num / 2))]
            # test_index = test_valid_index[int(round(batch_num / 2)):]
            if k == 0:
                train_index = index_ls[batch_num:]
            else:
                train_index = index_ls[0:k*batch_num] + index_ls[(k+1)*batch_num:]

            train_x.append(x[train_index])
            train_y.append(y[train_index])
            # valid_x.append(x[valid_index])
            # valid_y.append(y[valid_index])
            test_x.append(x[test_index])
            test_y.append(y[test_index])

        return train_x, train_y, test_x, test_y

    def make_test_dataset(self, x, y, target_ts_dict, save_dir, output_dim, test_num=10, k_folds=5, rand=False, shuffle=False):
        """
        実験の再現性確保のため, kfoldのcvを複数回行うデータセットを全て保存しておく.
        """

        # for test_index in xrange(test_num):
        #     #
        #     # os.mkdir(save_dir + '%s_test' % test_index)
        #     if shuffle:
        #         train_x, train_y, test_x, test_y = self.make_shuffle_k_folds_data(x, y, k_folds, rand)
        #
        #         for k in xrange(k_folds):
        #             fold_dir = save_dir + '%s_test/%s_fold/' % (test_index, k)
        #             os.mkdir(fold_dir)
        #             numpy.savez(fold_dir + '%s_train.npz' % k, x=train_x[k], y=train_y[k])
        #             numpy.savez(fold_dir + '%s_test.npz' % k, x=test_x[k], y=test_y[k])
        #     else:
        train_x, test_x = x['train'], x['test']
        train_y, test_y = y['train'], y['test']
        train_target_ts, test_target_ts = target_ts_dict['train'], target_ts_dict['test']

        type_dir = save_dir + 'output_dim=%s/' % output_dim
        os.mkdir(type_dir)
        numpy.savez(type_dir + 'train.npz', x=train_x, y=train_y)
        numpy.savez(type_dir + 'test.npz', x=test_x, y=test_y)
        numpy.savez(type_dir + 'target.npz', train=train_target_ts, test=test_target_ts)

    def balancing_sampling(self, x, label):

        balanced_x_dict, balanced_label_dict = {}, {}
        for data_type in ['train', 'test']:

            label_0_ls, label_1_ls = [], []
            # label0, 1を分ける
            for x_ele, label_ele in zip(x, label):

                if label_ele == 1:
                    label_1_ls.append(x_ele)
                else:
                    label_0_ls.append(x_ele)

            balanced_x, balanced_label = [], []
            for i in xrange(len(x)):
                label = numpy.random.choice([0, 1])
                balanced_label.append(label)

                if label == 0:
                    index = numpy.random.randint(len(label_0_ls))
                    balanced_x.append(label_0_ls[index])
                else:
                    index = numpy.random.randint(len(label_1_ls))
                    balanced_x.append(label_1_ls[index])

            balanced_x_dict.update({data_type: balanced_x})
            balanced_label_dict.update({data_type: balanced_label})

        return balanced_x_dict, balanced_label_dict
