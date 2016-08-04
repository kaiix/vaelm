#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import sys
import cPickle as pickle

from matplotlib import pyplot as plt
import numpy as np


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
    for k in merged_data:
        merged_data[k] = sorted(merged_data[k], key=lambda v: v[0])
    return merged_data


def average_sample(x, step):
    return [np.mean(x[i:i + step]) for i in xrange(0, len(x), step)]


def main(data_dir):
    plt.figure()

    # plot overall loss
    plt.subplot(2, 1, 1)
    data = merge_metadata(data_dir)
    train_loss = data['train_loss']
    steps, losses = zip(*train_loss)
    plt.plot(steps, losses, label='train loss')
    dev_loss = data['dev_loss']
    steps, losses = zip(*dev_loss)
    plt.plot(steps, losses, color='r', label='dev loss')
    reconstruction_loss = data['reconstruction_loss']
    steps, losses = zip(*reconstruction_loss)
    plt.plot(steps, losses, color='g', label='reconstruction loss')
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.legend(loc='upper left')
    plt.ylim((0, 10))

    # plot vae loss, and annealing weight
    ax1 = plt.subplot(2, 1, 2)
    annealing_weight = data['annealing_weight']
    steps, weights = zip(*annealing_weight)
    ax1.plot(steps, weights, 'b-', label='KL term weight', lw=2.0)
    plt.ylim(0, 1)
    plt.ylabel('KL term weight')
    plt.xlabel('step')

    ax2 = ax1.twinx()
    kl_loss = data['kl_loss']
    steps, losses = zip(*kl_loss)
    ax2.plot(steps, losses, 'r-', label='KL term value', lw=2.0)
    plt.ylim(0, 8)
    plt.yticks(np.linspace(0, 8, 9))
    plt.ylabel('KL term value')

    plt.show()


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        print("""usage:
        {} log_dir
        """.format(__file__))
        sys.exit()

    main(args[0])
