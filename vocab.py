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


def error(msg):
    raise Exception(msg)


class Vocab(object):
    def __init__(self, path):
        self.size = 0
        self._index = {}
        self._tokens = {}
        self.unk_index = None
        self.unk_token = None
        self.start_index = None
        self.start_token = None
        self.end_index = None
        self.end_token = None
        self.pad_index = None
        self.pad_token = None

        with open(path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                token = line.strip()
                self._tokens[self.size] = token
                self._index[token] = self.size
                self.size += 1

        unks = ['<unk>', '<UNK>']
        for tok in unks:
            self.unk_index = self.unk_index or self._index.get(tok)
            if self.unk_index is not None:
                self.unk_token = tok
                break

        starts = ['<go>', '<GO>']
        for tok in starts:
            self.start_index = self.start_index or self._index.get(tok)
            if self.start_index is not None:
                self.start_token = tok
                break

        ends = ['<eos>', '<EOS>']
        for tok in ends:
            self.end_index = self.end_index or self._index.get(tok)
            if self.end_index is not None:
                self.end_token = tok
                break

        pads = ['<pad>', '<PAD>']
        for tok in pads:
            self.pad_index = self.pad_index or self._index.get(tok)
            if self.pad_index is not None:
                self.pad_token = tok
                break

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
            if not self.unk_index:
                error(
                    'Token not in vocabulary and no UNK token defined: {}'.format(
                        w))
            return self.unk_index
        return self._index[w]

    def token(self, i):
        if i < 0 or i > self.size:
            error('Index {} out of bounds'.format(i))
        return self._tokens[i]

    def map(self, tokens):
        output = map(self.index, tokens)
        return output

    def add_unk_token(self):
        if self.unk_token is not None:
            return
        self.unk_index = self.add('<unk>')

    def add_start_token(self):
        if self.start_token is not None:
            return
        self.start_index = self.add('<s>')

    def add_end_token(self):
        if self.end_token is not None:
            return
        self.end_index = self.add('</s>')

    def add_pad_token(self):
        if self.pad_token is not None:
            return
        self.pad_index = self.add('<pad>')


if __name__ == '__main__':
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/sick/'
    vocab = Vocab(data_dir + 'vocab-cased.txt')
    print('pad:', vocab.pad_index, vocab.pad_token)
    print('unk:', vocab.unk_index, vocab.unk_token)
    print('start:', vocab.start_index, vocab.start_token)
    print('end:', vocab.end_index, vocab.end_token)
