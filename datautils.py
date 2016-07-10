"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os


class Metadata(object):
    def __init__(self, save_dir):
        assert os.path.exists(save_dir)
        self._save_dir = save_dir
        self._data = {}
        self._global_step = -1

    def add(self, global_step, key, value):
        if key not in self._data:
            self._data[key] = []
        self._data[key].append((global_step, value))
        self._global_step = max(global_step, self._global_step)

    def save(self):
        assert self._global_step > 0
        import cPickle as pickle
        filename = os.path.join(self._save_dir,
                                'metadata.{}.pkl'.format(self._global_step))
        with open(filename, 'wb') as f:
            pickle.dump(self._data, f)

            # clean
            for k in self._data:
                self._data[k] = []
