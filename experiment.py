from __future__ import print_function
import os
import random
import math
import numpy
import theano.tensor as T
from opendeep.data import Dataset, TRAIN, VALID
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
    def __init__(self, source_dir, size_key, one_hot=False, train_split=0.9):
        print("Getting dataset %s" % size_key)
        # grab the datasets from the preprocessed files
        datasets = [(numpy.load(f), get_label(f)) for f in find_processed_files(source_dir, size_key)]
        # make sure they are all the correct dimensionality
        datasets = [(data.shape, data, label) for data, label in datasets
                    if data.shape[1] == sizes[size_key] and label is not None]

        print("Found %d examples" % len(datasets))

        # shuffle!
        random.shuffle(datasets)

        # data
        dataset = [dataset_shared(data, borrow=True) for _, data, _ in datasets]
        # labels
        labels = numpy.asarray([label for _, _, label in datasets], dtype='int8')
        # make the labels into one-hot vectors
        if one_hot:
            labels = numpy_one_hot(labels, n_classes=5)
        # shapes
        shapes = (numpy.sum([shape for shape, _, _ in datasets], axis=0)[0], sizes[size_key])
        print('Overall shape: %s' % str(shapes))

        train_len = int(math.floor(train_split * len(dataset)))
        print("# train: %d examples" % train_len)
        self.train = (dataset_shared(dataset[:train_len]), dataset_shared(labels[:train_len]))
        self.valid = (dataset_shared(dataset[train_len:]), dataset_shared(labels[train_len:]))
        self.train_shapes = shapes[:train_len]
        self.valid_shapes = shapes[train_len:]
        print("Dataset %s initialized!" % size_key)

        s = self.train[0]
        print(s[0].eval())

    def getSubset(self, subset):
        if subset is TRAIN:
            return self.train
        elif subset is VALID:
            return self.valid
        else:
            return None, None

    def getDataShape(self, subset):
        if subset is TRAIN:
            return self.train_shapes
        elif subset is VALID:
            return self.valid_shapes
        else:
            return None


def main(size_key):

    out_vector = True

    # grab the data for this step size, window, and max frequency
    heartbeats = HeartSound(basedir, size_key, one_hot=out_vector)
    return

    # define our model! we are using lstm with mean-pooling and softmax as classification
    hidden_size = 128
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
                      noise_level=0.3)

    # mean of the hiddens across timesteps (reduces ndim by 1)
    mean_pooling = T.mean(lstm_layer.get_hiddens(), axis=0)

    # now the classification layer
    softmax_layer = SoftmaxLayer(inputs_hook=(hidden_size, mean_pooling),
                                 output_size=n_classes,
                                 out_as_probs=out_vector)

    # make it into a prototype!
    model = Prototype(layers=[lstm_layer, softmax_layer])


    # optimizer
    optimizer = RMSProp(dataset=heartbeats,
                        model=model,
                        n_epoch=300,
                        batch_size=1,
                        save_frequency=10)

    # monitors
    errors = Monitor(name='error', expression=model.get_monitors()['softmax_error'], train=True, valid=True)

    # plot the monitor
    plot = Plot('heartbeat %s' % size_key, monitor_channels=[errors], open_browser=True)

    optimizer.train(plot=plot)



if __name__ == '__main__':
    # if we want logging
    config_root_logger()

    step = 10
    window = 20
    max_freq = 4000

    size_key = "%d_%d_%d" % (step, window, max_freq)
    main(size_key)