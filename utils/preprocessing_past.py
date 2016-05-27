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
    data: Input data. Must be pandas object.
    k_folds: Number of folds of cross validation. Must be at least 2.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self, path):

        data = open(self.data_dir + path)

        return data

    def calc_bookmark_num(self, data):

        vol_ls = []

        for i, line in enumerate(data):
            vol = int(line.split(" ")[0])
            if i != 0:
                vol_ls.append(vol)

        return vol_ls

    def timeseries(self, data, sum_min=100):

        list, category_ts, category_sum = [], [], []

        for i, line in enumerate(data):
            if i == 0:
                continue

            elif 0 < i < 100001:
                # if i % 1000 == 0:
                    # sys.stdout.write("\r%d web page ended" % i)
                    # sys.stdout.flush()

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

    def make_started_ts(self, category_ts):

        category_started_ts = []

        for i, ts in enumerate(category_ts):

            # if i % 100 == 0:
                # sys.stdout.write("\r%d" % i)
                # sys.stdout.flush()

            start_flag = 0 # bookmarkが開始したかフラグ
            started_ts = [] # bookmark開始以降の時系列を取得

            for day_b in ts:

                if day_b > 0:
                    start_flag = 1
                # bookmarkが開始されていれば, それ以降の時系列を60日ずつ取得.
                # 30 - 1, 30 - 3, 30 - 7, 30 - 30に分割し学習データとする.

                if start_flag:
                    started_ts.append(day_b)
            category_started_ts.append(started_ts)

        return category_started_ts

    def make_supervised_data(self, category_started_ts, stride=30, input_dim=30):

        category_input_ts, category_label, category_input_sum = [], [], []
        target_dim = 2 * input_dim
        data_num = 0

        for started_ts in category_started_ts:

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

                        input_ts = target_ts[:input_dim]
                        output_ts = target_ts[input_dim:]
                        output_average = numpy.average(output_ts)

                        ## 二値分類
                        if input_average < output_average: label = 1.0
                        else: label = 0.0

                        category_input_ts.append(input_ts)
                        category_label.append(label)
                        # inputの総bookmark数はいらないけどモデルを分ける要因である可能性があるので解析のため残す
                        category_input_sum.append(numpy.array([input_sum]))

                        data_num += 1
                        # sys.stdout.write("\r%d data created" % data_num)
                        # sys.stdout.flush()

        return category_input_ts, category_label, category_input_sum

    def make_k_folds_data(self, x, y, k_folds=5):

        batch_num = len(x) / k_folds

        random_index = range(len(x))
        random.shuffle(random_index)

        x, y = numpy.array(x), numpy.array(y)
        train_x, train_y, test_x, test_y = [], [], [], []

        for k in xrange(k_folds):
            test_index = random_index[k*batch_num:(k+1)*batch_num]
            if k == 0:
                train_index = random_index[batch_num:]
            else:
                train_index = random_index[0:k*batch_num] + random_index[(k+1)*batch_num:]

            train_x.append(x[train_index])
            train_y.append(y[train_index])
            test_x.append(x[test_index])
            test_y.append(y[test_index])

        return train_x, train_y, test_x, test_y


    def make_test_dataset(self, x, y, save_dir, test_num=10, k_folds=5):
        """
        実験の再現性確保のため, kfoldのcvを複数回行うデータセットを全て保存しておく.
        """

        for test_index in xrange(test_num):
            os.mkdir(save_dir + '%s_test' % test_index)
            train_x, train_y, test_x, test_y = self.make_k_folds_data(x, y, k_folds)

            for k in xrange(k_folds):
                fold_dir = save_dir + '%s_test/%s_fold/' % (test_index, k)
                os.mkdir(fold_dir)
                numpy.savez(fold_dir + '%s_train.npz' % k, x=train_x[k], y=train_y[k])
                numpy.savez(fold_dir + '%s_test.npz' % k, x=test_x[k], y=test_y[k])

    def balancing_sampling(self, x, label):

        label_0_ls, label_1_ls = [], []

        ## label0, 1を分ける
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

        return balanced_x, balanced_label