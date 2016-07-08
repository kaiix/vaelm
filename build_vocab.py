#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import sys
import collections

import tensorflow as tf

_START_VOCAB = ['<pad>', '<unk>', '<go>', '<eos>']


def build_vocab(filepaths, dst_path, lowercase=True, max_vocab_size=40000):
    # TODO: unicode text
    vocab = []
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab.extend(line.split())
    vocab = _START_VOCAB + sorted(set(vocab))
    with open(dst_path, 'w') as f:
        for w in vocab:
            f.write(w + '\n')


def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return words, word_to_id


def main(data_path):
    assert os.path.exists(data_path)
    vocab_path = os.path.join(os.path.dirname(data_path), 'vocab-cased.txt')

    words, word_to_id = _build_vocab(data_path)
    extra_tokens = []
    for tok in _START_VOCAB:
        if tok not in word_to_id:
            extra_tokens.append(tok)
    vocab = extra_tokens + list(words)

    with open(vocab_path, 'w') as f:
        for w in vocab:
            f.write(w+'\n')

    return vocab


if __name__ == '__main__':
    print('Building vocab for dataset')
    vocab = main(sys.argv[1])
    print('vocab size = {}'.format(len(vocab)))
