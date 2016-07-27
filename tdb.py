from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import tensorflow as tf

__VERSION__ = '0.0.1'


def print_version():
    print(__VERSION__)


def tvars(name=None):
    if not name:
        for v in tf.trainable_variables():
            print(v.name)
        return

    ref = tf.get_default_graph().get_tensor_by_name(name)
    return ref.eval()
