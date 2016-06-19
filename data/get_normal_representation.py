# coding:utf-8
import numpy
import six
import pandas as pd
import random
import os
import sys
from base import BaseDataManager


def get_ts_normal_representation(data, sum_min=100):
    """
    :param data: Opened text object of data.
    :param sum_min: Threshold of bookmark volume.
    :return category_ts: List object of timeseries data divided by category.
    :return category_sum: List object of boookmark volume of each pages devided by category. (not used)
    """
    event_ls, ts_ls, ts_sum_ls = [], [], []

    for i, line in enumerate(data):
        if i == 0:
            continue

        elif 0 < i < 500001:
            # if i % 1000 == 0:
            #     sys.stdout.write("\r%d web page ended" % i)
            #     sys.stdout.flush()

            event_ls.append(line.split(" "))
            ts_sum = int(event_ls[i-1][0])

            if ts_sum > sum_min:
                time_series = [0 for j in six.moves.range(1600)]

                for event in event_ls[i - 1][1:]:
                    day = int(event.split(":")[1])
                    time_series[day] += 1

                ts_ls.append(numpy.array(time_series))
                ts_sum_ls.append(ts_sum)

        else:
            break

    return ts_ls, ts_sum_ls


def calc_bookmark_num(data):
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
