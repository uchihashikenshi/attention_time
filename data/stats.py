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


class Stats(BaseDataManager):

    def __init__(self, raw_data_dir, save_data_dir):
        super(Stats, self).__init__(
            raw_data_dir=raw_data_dir,
            save_data_dir=save_data_dir
        )

    def pattern_analysis(self):
        pass
