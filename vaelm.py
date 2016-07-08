"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import time
import sys

import numpy as np
import tensorflow as tf

from vocab import Vocab
from model import VariationalAutoEncoder

flags = tf.flags

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
flags.DEFINE_integer('batch_size', 25, 'Batch size to use during training.')
flags.DEFINE_integer('num_units', 150, 'Size of each LSTM layer.')
flags.DEFINE_integer('embedding_size', 300, 'Size of word embedding.')
flags.DEFINE_string('data_dir', './data/sick', 'Data directory')
flags.DEFINE_string('train_dir', './checkpoints', 'Training directory.')
flags.DEFINE_string('log_dir', './logs', 'Log directory.')
flags.DEFINE_integer('max_train_data_size', 0,
                     'Limit on the size of training data (0: no limit).')
flags.DEFINE_integer('steps_per_checkpoint', 200,
                     'How many training steps to do per checkpoint.')
flags.DEFINE_boolean('save', False, 'Save checkpoint files.')
flags.DEFINE_boolean('eval', False, 'Run a evaluation process.')
flags.DEFINE_boolean('use_embedding', False, 'Use pre-trained embedding')
flags.DEFINE_string('embedding', './data/sick/sick.300d.npy',
                    'Pre-trained word embeddings')

FLAGS = flags.FLAGS

_buckets = [10, 15, 20, 25, 30]


def read_data(root_dir, vocab, max_size=None):
    print('load data from "{}"'.format(root_dir))
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(root_dir + 'a.toks') as source_file:
        counter = 0
        while not max_size or counter < max_size:
            source = source_file.readline()
            if not source:
                break
            counter += 1

            source_ids = map(vocab.index, source.split())

            for bucket_id, seq_length in enumerate(_buckets):
                if len(source_ids) < seq_length:
                    data_set[bucket_id].append(source_ids)
                    break

    for i, b in enumerate(_buckets):
        print('bucket {} ({}) has {} sentences'.format(i, b, len(data_set[i])))
    print('total bucket size = {}'.format(sum(map(len, data_set))))

    return data_set


def create_model(sess, vocab, forward_only=False):
    model = VariationalAutoEncoder(FLAGS.learning_rate, FLAGS.batch_size,
                                   FLAGS.num_units, FLAGS.embedding_size,
                                   FLAGS.max_gradient_norm, _buckets, vocab,
                                   forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print('Reading model parameters from {}'.format(
            ckpt.model_checkpoint_path))
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        sess.run(tf.initialize_all_variables())
    return model


def train():
    vocab = Vocab(os.path.join(FLAGS.data_dir, 'vocab-cased.txt'))
    print('Reading development and training data (limit: {}).'.format(
        FLAGS.max_train_data_size))

    dev_set = read_data(
        os.path.join(FLAGS.data_dir, 'dev/'), vocab, FLAGS.max_train_data_size)
    train_set = read_data(
        os.path.join(FLAGS.data_dir, 'train/'), vocab,
        FLAGS.max_train_data_size)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    metadata = Metadata(FLAGS.log_dir)

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            model = create_model(sess, vocab, False)
            if FLAGS.use_embedding:
                print('Loading word embeddings')
                emb_vecs = np.load(FLAGS.embedding)
                print('Using pre-trained word embedding')
                sess.run(model.embedding.assign(emb_vecs))

        step_time, loss = 0.0, 0.0
        current_step = 0
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            # print_data(encoder_inputs, decoder_inputs, target_weights, vocab)
            step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                      target_weights, bucket_id, False)
            step_time += \
                (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            global_step = model.global_step.eval()
            metadata.add(global_step, 'train_loss', step_loss)

            if current_step % 10 == 0:
                dev_loss = sampled_loss(sess, model, dev_set)
                metadata.add(global_step, 'dev_loss', dev_loss)

            if current_step % 1000 == 0:
                print('  saving metadata ...')
                metadata.save()

            if current_step % FLAGS.steps_per_checkpoint == 0:
                print('global step {} step-time {:.2f} loss {:f}'
                      .format(model.global_step.eval(), step_time, loss))
                step_time, loss = 0.0, 0.0

                if FLAGS.save:
                    print('  saving checkpoint ...')
                    sys.stdout.flush()
                    checkpoint_path = os.path.join(FLAGS.train_dir,
                                                   "model.ckpt")
                    model.saver.save(sess,
                                     checkpoint_path,
                                     global_step=model.global_step)

            if current_step % 10000 == 0:
                break


def print_data(batch_encoder_inputs, batch_decoder_inputs,
               batch_target_weights, vocab):
    assert len(batch_encoder_inputs) == len(batch_decoder_inputs) - 1
    assert len(batch_target_weights) == len(batch_decoder_inputs)
    # print('PAD: ', vocab.pad_index, 'EOS', vocab.end_index, 'UNK',
    #       vocab.unk_index)
    batch_size = batch_encoder_inputs[0].shape[0]
    encoder_inputs = [[] for _ in xrange(batch_size)]
    for length_idx in xrange(len(batch_encoder_inputs)):
        for batch_idx in xrange(len(batch_encoder_inputs[length_idx])):
            encoder_inputs[batch_idx].append(batch_encoder_inputs[length_idx][
                batch_idx])
    decoder_inputs = [[] for _ in xrange(batch_size)]
    for length_idx in xrange(len(batch_decoder_inputs)):
        for batch_idx in xrange(len(batch_decoder_inputs[length_idx])):
            decoder_inputs[batch_idx].append(batch_decoder_inputs[length_idx][
                batch_idx])
    target_weights = [[] for _ in xrange(batch_size)]
    for length_idx in xrange(len(batch_target_weights)):
        for batch_idx in xrange(len(batch_target_weights[length_idx])):
            target_weights[batch_idx].append(batch_target_weights[length_idx][
                batch_idx])
    for enc, dec, w in zip(encoder_inputs, decoder_inputs, target_weights):
        print('encoder input > "{}"'.format(map(vocab.token, enc)))
        print('decoder input > "{}"'.format(map(vocab.token, dec)))
        print('target weights > "{}"'.format(list(zip(
            map(vocab.token, dec[1:]), w))))


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


def sampled_loss(session, model, dev_set):
    dev_loss = 0.0
    nbuckets = 0
    for bucket_id in xrange(len(_buckets)):
        if len(dev_set[bucket_id]) == 0:
            print('  eval: empty bucket {}'.format(bucket_id))
            continue

        nbuckets += 1

        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            dev_set, bucket_id)
        eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                  target_weights, bucket_id, True)
        print('  eval: bucket {} loss {:f}'.format(bucket_id, eval_loss))
        dev_loss += eval_loss
    dev_loss /= nbuckets
    print('  average sampled dev loss: {:f}'.format(dev_loss))
    return dev_loss


def evaluate():
    vocab = Vocab(os.path.join(FLAGS.data_dir, 'vocab-cased.txt'))

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            model = create_model(sess, vocab, True)
            model.batch_size = 1

        while True:
            source = raw_input('> ')
            output = model.predict(sess, source)
            print('predict result: {}'.format(output))


def main(_):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if FLAGS.eval:
        evaluate()
    else:
        train()


if __name__ == '__main__':
    tf.app.run()
