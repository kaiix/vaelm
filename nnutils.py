from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import collections

import tensorflow as tf
import numpy as np


def _is_sequence(seq):
    return (isinstance(seq, collections.Sequence) and
            not isinstance(seq, basestring))


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (_is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not _is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear is expecting 2D arguments: %s" %
                             str(shapes))
        if not shape[1]:
            raise ValueError("Linear expects shape[1] of arguments: %s" %
                             str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term


def same_shape(s1, s2):
    def _as_tuple(s):
        if isinstance(s, tf.Tensor):
            return tuple(s.get_shape().as_list())
        if isinstance(s, np.ndarray):
            return s.shape
        return tuple(s)

    return _as_tuple(s1) == _as_tuple(s2)
