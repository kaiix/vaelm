#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import sys
import cPickle as pickle

from matplotlib import pyplot as plt


def listdir(root_dir, predicate=lambda f: True):
    result = []
    for (p, d, filenames) in os.walk(root_dir):
        if not filenames:
            continue
        result.extend([os.path.join(p, f) for f in filenames if predicate(f)])
    return result


def merge_metadata(data_dir):
    merged_data = {}
    for filename in listdir(data_dir, lambda f: f.endswith('.pkl')):
        with open(filename) as pkl:
            part = pickle.load(pkl)
            for k, v in part.iteritems():
                if k in merged_data:
                    merged_data[k].extend(v)
                else:
                    merged_data[k] = v
            print('merged {}'.format(filename))
    return merged_data


def main(data_dir):
    data = merge_metadata(data_dir)
    train_loss = data['train_loss']
    steps, losses = zip(*train_loss)
    plt.plot(steps, losses, label='train loss')
    dev_loss = data['dev_loss']
    steps, losses = zip(*dev_loss)
    plt.plot(steps,
             losses,
             marker='o',
             linestyle='--',
             color='r',
             label='dev loss')
    plt.xlabel('step')
    plt.ylabel('train_loss')
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
