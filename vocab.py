#!/usr/bin/env python
'''
A vocabulary object. Initialized from a file with one vocabulary token per
line. Maps between vocabulary tokens and indices. If an UNK token is defined in
the vocabulary, returns the index to this token if queried for an
out-of-vocabulary token.
'''
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function


class Vocab(object):
    def __init__(self, path, max_vocab=40000, add_special=True):
        self.size = 0
        self._index = {}
        self._tokens = {}

        if add_special:
            self.unk_token = '<unk>'
            self.unk_index = self.add(self.unk_token)
            self.eos_token = '<eos>'
            self.eos_index = self.add(self.eos_token)
            self.pad_token = '<pad>'
            self.pad_index = self.add(self.pad_token)
        else:
            self.unk_token, self.unk_index = None, None
            self.eos_token, self.eos_index = None, None
            self.pad_token, self.pad_index = None, None

        with open(path) as f:
            while self.size < max_vocab:
                line = f.readline()
                if not line:
                    break
                token = line.strip()
                self.add(token)

    def contains(self, w):
        return w in self._index

    def add(self, w):
        if w in self._index:
            return self._index[w]
        self._tokens[self.size] = w
        self._index[w] = self.size
        self.size = self.size + 1
        return self.size - 1

    def index(self, w):
        if w not in self._index:
            if self.unk_index is None:
                raise ValueError(
                    'Token not in vocabulary and no UNK token defined: {}'.format(
                        w))
            return self.unk_index
        return self._index[w]

    def token(self, i):
        if i < 0 or i > self.size:
            raise ValueError('Index {} out of bounds'.format(i))
        return self._tokens[i]


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/ptb/'
    vocab = Vocab(data_dir + 'vocab-cased.txt', 100)
    print('unk:', vocab.unk_index, vocab.unk_token)
    print('end:', vocab.eos_index, vocab.eos_token)
    print('pad:', vocab.pad_index, vocab.pad_token)
    print('size = ', vocab.size)
