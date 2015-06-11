from __future__ import print_function
import os
import struct
import random
import math
import numpy
import theano.tensor as T
from opendeep.data import Dataset, TRAIN, VALID, TEST
from opendeep.log import config_root_logger
from opendeep.models.single_layer.lstm import LSTM
from opendeep.optimization import RMSProp
from opendeep.monitor import Monitor, Plot
from opendeep import dataset_shared

basedir = 'data/ecg/apnea-ecg'
label_ext = ".apn"
data_ext = ".dat"
extra_ext = ".qrs"


def find_train_files(directory, find_ext):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            name, ext = os.path.splitext(basename)
            if ext == find_ext and name.split(os.sep)[-1][0] in ['a', 'b', 'c'] and len(name.split(os.sep)[-1]) == 3:
                filename = os.path.join(root, basename)
                yield filename

class ECG(Dataset):
    def __init__(self, source_dir, train_split=0.8, valid_split=0.15):
        self.train = None, None
        self.train_shape = None
        self.valid = None, None
        self.valid_shape = None
        self.test = None, None
        self.test_shape = None


    def getSubset(self, subset):
        if subset is TRAIN:
            return self.train
        elif subset is VALID:
            return self.valid
        elif subset is TEST:
            return self.test
        else:
            return None, None

    def getDataShape(self, subset):
        if subset is TRAIN:
            return self.train_shape
        elif subset is VALID:
            return self.valid_shape
        elif subset is TEST:
            return self.test_shape
        else:
            return None


def main():
    pass



if __name__ == '__main__':
    # http://www.physionet.org/physiobank/database/apnea-ecg/

    # if we want logging
    config_root_logger()

    i=0
    for f in find_train_files(basedir, label_ext):
        if i==0:
            data = numpy.fromfile(f, dtype=numpy.bool)
            print(data.shape)
        else:
            pass
        i+=1

    i=0
    for f in find_train_files(basedir, data_ext):
        if i == 0:
            data = numpy.fromfile(f, dtype=numpy.float16)
            print(data.shape)
        else:
            pass
        i += 1

    i = 0
    for f in find_train_files(basedir, extra_ext):
        if i == 0:
            data = numpy.fromfile(f, dtype=numpy.bool)
            print(data.shape)
        else:
            pass
        i += 1