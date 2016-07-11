# coding:utf-8
from __future__ import print_function
import numpy
import pandas as pd
import six
import random
import os


class BaseDataManager(object):

    def __init__(self, raw_data_dir, save_data_dir):
        self.raw_data_dir = raw_data_dir
        self.save_data_dir = save_data_dir

    # def get_raw_data_home(self, data_home=None):
    #     try:
    #         data_home = os.path.expanduser(data_home)
    #     except NameError:
    #         print "data home is not existed"
    #
    #     if not os.path.exists(data_home):
    #         raise NameError

    def load_raw_data(self, path):
        """
        :param path: Data path.
        :return: data: Loaded data.
        """
        data = open(self.raw_data_dir + path)

        return data

    def load_normal_representation_data(self, output_dim, data_type='balance', category_name='ajax'):
        """
        :param data_type:
        :param category_name:
        :param output_dim:
        :return:
        """
        save_data_dir = '%s%s/' % (self.save_data_dir, data_type)
        train = numpy.load(save_data_dir + category_name + '/output_dim=%s/train.npz' % output_dim)
        test = numpy.load(save_data_dir + category_name + '/output_dim=%s/test.npz' % output_dim)
        target = numpy.load(save_data_dir + category_name + '/output_dim=%s/target.npz' % output_dim)

        return train, test, target

    def div_train_test(self, ts_ls, train_test_ratio, div_type='web'):
        """
        :param ts_ls: List object of timeseries data divided by category.
        :param train_test_ratio: Ratio of train data. Must be [0, 1] continuous value.
        :param div_type:
        :return ts_dict: Dict of timeseries divided by categories. Key is train and test.
        """
        ts_dict = {}
        if div_type == 'web':
            random_ind = range(len(ts_ls))
            random.shuffle(random_ind)

            train_ind = random_ind[:int(len(ts_ls) * train_test_ratio)]
            test_ind = random_ind[int(len(ts_ls) * train_test_ratio):]

            train_ts = numpy.array(ts_ls)[train_ind]
            test_ts = numpy.array(ts_ls)[test_ind]

            ts_dict.update({'train': train_ts})
            ts_dict.update({'test': test_ts})
        else:
            raise NotImplementedError()

        return ts_dict

    @staticmethod
    def get_started_ts(ts_dict):
        started_ts_dict = {}
        for data_type in ['train', 'test']:

            started_ts_ls = []
            for i, ts in enumerate(ts_dict[data_type]):

                # if i % 100 == 0:
                #     sys.stdout.write("\r%d" % i)
                #     sys.stdout.flush()

                start_flag = 0
                started_ts = []
                for ts_ele in ts:

                    if ts_ele > 0:
                        start_flag = 1

                    if start_flag:
                        started_ts.append(ts_ele)


                started_ts_ls.append(started_ts)

            started_ts_dict.update({data_type: started_ts_ls})

        return started_ts_dict

    @staticmethod
    def make_supervised_data(started_ts_dict, stride, output_dim, input_average_min=2, input_dim=30):
        input_ts_dict, label_dict, target_ts_dict, input_sum_dict = {}, {}, {}, {}
        for data_type in ['train', 'test']:

            input_ts_ls, label_ls, target_ts_ls, input_sum_ls = [], [], [], []
            target_dim = 2 * input_dim

            label_dim = input_dim / output_dim
            if input_dim % output_dim != 0:
                print("label_dim=%s: error" % label_dim)
            data_num = 0
            for started_ts_ls in started_ts_dict[data_type]:

                for i in six.moves.range(200):

                    # stackoverflow阻止
                    if len(started_ts_ls) < target_dim + stride * i:
                        break
                    else:
                        target_ts_array = numpy.array(started_ts_ls[stride * i:target_dim + stride * i])
                        input_average = numpy.average(target_ts_array[:input_dim])
                        input_sum = numpy.sum(numpy.array(target_ts_array[:input_dim]))

                        if input_average > input_average_min:

                            label = []
                            input_ts = target_ts_array[:input_dim]
                            for dim in six.moves.range(output_dim):
                                output_ts = target_ts_array[input_dim + label_dim * dim:input_dim + label_dim * (dim + 1)]
                                output_average = numpy.average(output_ts)

                                if input_average < output_average:
                                    label_ele = 1.0
                                else:
                                    label_ele = 0.0
                                label.append(label_ele)

                            input_ts_ls.append(input_ts)
                            label_ls.append(label)
                            target_ts_ls.append(target_ts_array)
                            # inputの総bookmark数はいらないけどモデルを分ける要因である可能性があるので解析のため残す
                            input_sum_ls.append(numpy.array([input_sum]))

                            data_num += 1
                            # sys.stdout.write("\r%d data created" % data_num)
                            # sys.stdout.flush()

            index_ls = range(len(input_ts_ls))
            random.shuffle(index_ls)

            input_ts_dict.update({data_type: numpy.array(input_ts_ls)[index_ls]})
            label_dict.update({data_type: numpy.array(label_ls)[index_ls]})
            target_ts_dict.update({data_type: numpy.array(target_ts_ls)[index_ls]})
            input_sum_dict.update({data_type: numpy.array(input_sum_ls)[index_ls]})

        return input_ts_dict, label_dict, target_ts_dict, input_sum_dict

    def make_dataset(self):
        pass



