# coding:utf-8
from __future__ import print_function
import numpy
import six
import pandas as pd
import random
import os
import sys
from base import BaseDataManager


class Preprocessing(BaseDataManager):
    """
    :param data: Input data. Must be pandas object.
    :param k_folds: Number of folds of cross validation. Must be at least 2.
    """

    def __init__(self, raw_data_dir, save_data_dir):
        super(Preprocessing, self).__init__(
            raw_data_dir=raw_data_dir,
            save_data_dir=save_data_dir
        )

    @staticmethod
    def balancing_sampling(x, label, target, output_dim):

        balanced_x_dict, balanced_label_dict, balanced_target_dict = {}, {}, {}
        for data_type in ['train', 'test']:

            balanced_x, balanced_label, balanced_target = [], [], []
            for dim in six.moves.range(output_dim):
                if abs(dim) == 0:
                    x_, label_, target_ = x[data_type], label[data_type], target[data_type]
                else:
                    x_, label_, target_ = balanced_x, balanced_label, balanced_target
                    balanced_x, balanced_label, balanced_target = [], [], []

                label_0_x_ls, label_1_x_ls = [], []
                label_0_label_ls, label_1_label_ls = [], []
                label_0_target_ls, label_1_target_ls = [], []
                for i, x_ele in enumerate(x_):

                    label_ele = label_[i]
                    target_ele = target_[i]

                    if abs(label_ele[dim] - 1) == 0:
                        label_1_x_ls.append(x_ele)
                        label_1_label_ls.append(label_ele)
                        label_1_target_ls.append(target_ele)
                    else:
                        label_0_x_ls.append(x_ele)
                        label_0_label_ls.append(label_ele)
                        label_0_target_ls.append(target_ele)

                for j in six.moves.range(len(x_)):

                    rand_label = numpy.random.choice([0, 1])
                    if abs(rand_label - 1) == 0:

                        if len(label_1_label_ls) < 0.0001:
                            break
                        index = numpy.random.randint(len(label_1_label_ls))

                        label_1_x_ele, label_1_label_ele, label_1_target_ele = \
                            label_1_x_ls[index], label_1_label_ls[index], label_1_target_ls[index]

                        label_1_x_ls, label_1_label_ls, label_1_target_ls = \
                            numpy.delete(label_1_x_ls, index, axis=0), \
                            numpy.delete(label_1_label_ls, index, axis=0), \
                            numpy.delete(label_1_target_ls, index, axis=0)

                        balanced_x.append(label_1_x_ele)
                        balanced_label.append(label_1_label_ele)
                        balanced_target.append(label_1_target_ele)
                    else:

                        if len(label_0_label_ls) < 0.0001:
                            break
                        index = numpy.random.randint(len(label_0_label_ls))

                        label_0_x_ele, label_0_label_ele, label_0_target_ele = \
                            label_0_x_ls[index], label_0_label_ls[index], label_0_target_ls[index]

                        label_0_x_ls, label_0_label_ls, label_0_target_ls = \
                            numpy.delete(label_0_x_ls, index, axis=0), \
                            numpy.delete(label_0_label_ls, index, axis=0), \
                            numpy.delete(label_0_target_ls, index, axis=0)

                        balanced_x.append(label_0_x_ele)
                        balanced_label.append(label_0_label_ele)
                        balanced_target.append(label_0_target_ele)

            balanced_x_dict.update({data_type: numpy.array(balanced_x)})
            balanced_label_dict.update({data_type: numpy.array(balanced_label)})
            balanced_target_dict.update({data_type: numpy.array(balanced_target)})

        return balanced_x_dict, balanced_label_dict, balanced_target_dict
