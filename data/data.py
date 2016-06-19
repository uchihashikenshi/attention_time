# coding:utf-8
from __future__ import print_function
import numpy
import six
import pandas as pd
import random
import os
import sys
import other
from base import BaseDataManager
sys.path.append('/Users/admin/Desktop/projects/attention_time/utils')


class Data(BaseDataManager):

    def __init__(self, raw_data_dir, save_data_dir):
        super(Data, self).__init__(
            raw_data_dir=raw_data_dir,
            save_data_dir=save_data_dir
        )

    # def make_shuffle_k_folds_data(self, x, y, k_folds=5, rand=False):
    #     """
    #     分割時にシャッフルするための関数. 現在は使用していない.
    #     :param x:
    #     :param y:
    #     :param k_folds:
    #     :param rand:
    #     :return:
    #     """
    #
    #     batch_num = len(x) / k_folds
    #     index_ls = range(len(x))
    #
    #     if rand:
    #         sys.stdout.write("\rrandomized")
    #         sys.stdout.flush()
    #         random.shuffle(index_ls)
    #
    #     x, y = numpy.array(x), numpy.array(y)
    #     train_x, train_y, valid_x, valid_y, test_x, test_y = [], [], [], [], [], []
    #
    #     for k in six.moves.range(k_folds):
    #         test_index = index_ls[k*batch_num:(k+1)*batch_num]
    #         # valid_index = test_valid_index[:int(round(batch_num / 2))]
    #         # test_index = test_valid_index[int(round(batch_num / 2)):]
    #         if k == 0:
    #             train_index = index_ls[batch_num:]
    #         else:
    #             train_index = index_ls[0:k*batch_num] + index_ls[(k+1)*batch_num:]
    #
    #         train_x.append(x[train_index])
    #         train_y.append(y[train_index])
    #         # valid_x.append(x[valid_index])
    #         # valid_y.append(y[valid_index])
    #         test_x.append(x[test_index])
    #         test_y.append(y[test_index])
    #
    #     return train_x, train_y, test_x, test_y

    def make_train_data(self, x, y, target_ts_dict, output_dim, data_type, data_name, test_num=10, k_folds=5, rand=False, shuffle=False):
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

        type_dir = self.save_data_dir + '/%s/%s/output_dim=%s/' % (data_type, data_name, output_dim)
        other.mkdirs(type_dir)
        numpy.savez(type_dir + 'train.npz', x=train_x, y=train_y)
        numpy.savez(type_dir + 'test.npz', x=test_x, y=test_y)
        numpy.savez(type_dir + 'target.npz', train=train_target_ts, test=test_target_ts)


def load_bookmark():
    pass


def load_tweet_hash_tag():
    pass


def load_memetracker():
    pass
