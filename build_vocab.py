#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import sys


def words(line):
    """Splits a line of text into tokens."""
    return line.strip().split()


def create_vocabulary(lines, max_vocab, min_count=5):
    """Reads text lines and generates a vocabulary."""
    lines.seek(0, os.SEEK_END)
    nbytes = lines.tell()
    lines.seek(0, os.SEEK_SET)

    vocab = {}
    for lineno, line in enumerate(lines, start=1):
        for word in words(line):
            vocab.setdefault(word, 0)
            vocab[word] += 1

        if lineno % 100000 == 0:
            pos = lines.tell()
            sys.stdout.write('\rComputing vocabulary: %0.1f%% (%d/%d)...' %
                             (100.0 * pos / nbytes, pos, nbytes))
            sys.stdout.flush()

    sys.stdout.write('\n')

    vocab = [(tok, n) for tok, n in vocab.iteritems() if n >= min_count]
    vocab.sort(key=lambda kv: (-kv[1], kv[0]))

    num_words = min(len(vocab), max_vocab)

    if not num_words:
        raise Exception('empty vocabulary')

    print('vocabulary contains %d tokens' % num_words)

    vocab = vocab[:num_words]
    return [tok for tok, n in vocab]


def main(data_path, max_vocab=40000):
    assert os.path.exists(data_path)
    vocab_path = os.path.join(os.path.dirname(data_path), 'vocab-cased.txt')

    with open(data_path) as fi:
        vocab = create_vocabulary(fi, max_vocab)

    with open(vocab_path, 'w') as fo:
        for w in vocab:
            fo.write(w + '\n')

    return vocab


if __name__ == '__main__':
    vocab = main(sys.argv[1])

    print('done!')
