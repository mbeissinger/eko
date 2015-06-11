from __future__ import print_function
import os
import random
import math
import numpy
import theano.tensor as T
from opendeep.data import Dataset, TRAIN, VALID, TEST
from opendeep.log import config_root_logger
from opendeep.models import Prototype, SoftmaxLayer
from opendeep.models.single_layer.lstm import LSTM
from opendeep.optimization import RMSProp
from opendeep.monitor import Monitor, Plot
from opendeep.utils.misc import numpy_one_hot
from opendeep import dataset_shared

basedir = 'data/sounds/'

sizes = {'10_10_2000': 11,
         '10_20_2000': 21,
         '10_10_4000': 21,
         '10_20_4000': 41,
         '10_10_8000': 41,
         '10_20_8000': 81}

classes = ['normal', 'murmer', 'extrahls', 'artifact', 'extrastole']

def find_processed_files(directory, size_key):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            name, ext = os.path.splitext(basename)
            if ext == ".npy" and "processed_%s" % size_key in name:
                filename = os.path.join(root, basename)
                yield filename

def get_label(filename):
    for i, c in enumerate(classes):
        if c in filename:
            return i
    return None

class HeartSound(Dataset):
    def __init__(self, source_dir, size_key, one_hot=False, train_split=0.9, valid_split=0.1):
        print("Getting dataset %s" % size_key)
        # grab the datasets from the preprocessed files
        datasets = [(numpy.load(f), get_label(f)) for f in find_processed_files(source_dir, size_key)]
        # make sure they are all the correct dimensionality
        datasets = [(data.shape, data, label) for data, label in datasets
                    if data.shape[1] == sizes[size_key] and label is not None]

        print("Found %d examples" % len(datasets))

        # shuffle!
        random.seed(1)
        random.shuffle(datasets)

        # shapes
        shapes = [shape for shape, _, _ in datasets]
        # data
        dataset = [data for _, data, _ in datasets]
        # labels
        labels = numpy.asarray([label for _, _, label in datasets], dtype='int8')
        # make the labels into one-hot vectors
        if one_hot:
            labels = numpy_one_hot(labels, n_classes=5)

        train_len = int(math.floor(train_split * len(dataset)))
        valid_len = int(math.floor(valid_split * len(dataset)))
        valid_stop = train_len + valid_len
        print("# train: %d examples" % train_len)

        train_datasets = dataset[:train_len]
        train_labels = labels[:train_len]

        if valid_len > 0:
            valid_datasets = dataset[train_len:valid_stop]
            valid_labels = labels[train_len:valid_stop]
        else:
            valid_datasets = []

        if train_len + valid_len < len(dataset):
            test_datasets = dataset[valid_stop:]
            test_labels = labels[valid_stop:]
        else:
            test_datasets = []

        # median_train_len = int(numpy.median(numpy.asarray(shapes[:train_len]), axis=0)[0])
        min_train_len = int(numpy.min(numpy.asarray(shapes[:train_len]), axis=0)[0])
        max_train_len = int(numpy.max(numpy.asarray(shapes[:train_len]), axis=0)[0])
        # train = numpy.array([data[:min_train_len] for data in train_datasets], dtype='float32')
        train = numpy.array(
            [numpy.pad(data, [(0, max_train_len-data.shape[0]), (0, 0)], mode='constant') for data in train_datasets],
            dtype='float32'
        )
        self.train_shape = train.shape
        self.train = (dataset_shared(train, borrow=True), dataset_shared(train_labels, borrow=True))

        if len(valid_datasets) > 0:
            min_valid_len = int(numpy.min(numpy.asarray(shapes[train_len:valid_stop]), axis=0)[0])
            max_valid_len = int(numpy.max(numpy.asarray(shapes[train_len:valid_stop]), axis=0)[0])
            # valid = numpy.array([data[:min_valid_len] for data in valid_datasets], dtype='float32')
            valid = numpy.array(
                [numpy.pad(data, [(0, max_valid_len-data.shape[0]), (0, 0)], mode='constant') for data in
                 valid_datasets],
                dtype='float32'
            )
            self.valid_shape = valid.shape
            self.valid = (dataset_shared(valid, borrow=True), dataset_shared(valid_labels, borrow=True))
        else:
            self.valid = None, None
            self.valid_shape = None

        if len(test_datasets) > 0:
            min_test_len = int(numpy.min(numpy.asarray(shapes[valid_stop:]), axis=0)[0])
            max_test_len = int(numpy.max(numpy.asarray(shapes[valid_stop:]), axis=0)[0])
            # valid = numpy.array([data[:min_test_len] for data in test_datasets], dtype='float32')
            test = numpy.array(
                [numpy.pad(data, [(0, max_test_len - data.shape[0]), (0, 0)], mode='constant') for data in
                 test_datasets],
                dtype='float32'
            )
            self.test_shape = test.shape
            self.test = (dataset_shared(test, borrow=True), dataset_shared(test_labels, borrow=True))
        else:
            self.test = None, None
            self.test_shape = None


        print("Train shape: %s" % str(self.train_shape))
        print("Valid shape: %s" % str(self.valid_shape))
        print("Test shape: %s" % str(self.test_shape))
        print("Dataset %s initialized!" % size_key)


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


def main(size_key):

    out_vector = True

    # grab the data for this step size, window, and max frequency
    heartbeats = HeartSound(basedir, size_key, one_hot=out_vector)

    # define our model! we are using lstm with mean-pooling and softmax as classification
    hidden_size = int(max(math.floor(sizes[size_key]*.4), 1))
    n_classes = 5

    lstm_layer = LSTM(input_size=sizes[size_key],
                      hidden_size=hidden_size,
                      output_size=1,  # don't care about output size
                      hidden_activation='tanh',
                      inner_hidden_activation='sigmoid',
                      weights_init='uniform',
                      weights_interval='montreal',
                      r_weights_init='orthogonal',
                      clip_recurrent_grads=5.,
                      noise='dropout',
                      noise_level=0.6,
                      direction='bidirectional')

    # mean of the hiddens across timesteps (reduces ndim by 1)
    mean_pooling = T.mean(lstm_layer.get_hiddens(), axis=0)

    # last hiddens as an alternative to mean pooling over time
    last_hiddens = lstm_layer.get_hiddens()[-1]

    # now the classification layer
    softmax_layer = SoftmaxLayer(inputs_hook=(hidden_size, mean_pooling),
                                 output_size=n_classes,
                                 out_as_probs=out_vector)

    # make it into a prototype!
    model = Prototype(layers=[lstm_layer, softmax_layer], outdir='outputs/prototype%s' % size_key)

    # optimizer
    optimizer = RMSProp(dataset=heartbeats,
                        model=model,
                        n_epoch=500,
                        batch_size=10,
                        save_frequency=10,
                        learning_rate=1e-4, #1e-6
                        grad_clip=None,
                        hard_clip=False
                        )

    # monitors
    errors = Monitor(name='error', expression=model.get_monitors()['softmax_error'], train=True, valid=True, test=True)

    # plot the monitor
    plot = Plot('heartbeat %s' % size_key, monitor_channels=[errors], open_browser=True)

    optimizer.train(plot=plot)



if __name__ == '__main__':
    # http://www.peterjbentley.com/heartchallenge/

    # if we want logging
    config_root_logger()

    for step in [10]:
        for freq in [8000]:
            for window in [20, 10]:
                size_key = "%d_%d_%d" % (step, window, freq)
                main(size_key)
