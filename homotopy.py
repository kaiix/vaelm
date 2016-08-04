"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import cPickle as pickle
import random
import logging

import numpy as np
import tensorflow as tf

from nnutils import same_shape
from nnutils import linear
from vocab import Vocab

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)1.1s %(asctime)s %(module)s:%(lineno)d] %(message)s')

flags = tf.flags

# data
flags.DEFINE_string('data_dir', './data/ptb', 'Data directory')
flags.DEFINE_integer('max_train_data_size', 0,
                     'Limit on the size of training data (0: no limit).')
flags.DEFINE_string('embedding', './data/sick/sick.300d.npy',
                    'Pre-trained word embeddings')
# model
flags.DEFINE_integer('num_units', 300, 'Size of each LSTM layer.')
flags.DEFINE_integer('embedding_size', 400, 'Size of word embedding.')
flags.DEFINE_boolean('use_embedding', False, 'Use pre-trained embedding')
flags.DEFINE_float('annealing_pivot', 3e4, 'Annealing pivot.')
# parameters
flags.DEFINE_float('learning_rate', 0.004, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
flags.DEFINE_integer('batch_size', 50, 'Batch size to use during training.')
flags.DEFINE_integer('max_steps', 5000,
                     'Number of (global) training steps to perform.')
flags.DEFINE_float('keep_prob', 1.0,
                   'keep probability for dropout regularization.')
flags.DEFINE_boolean('share_param', False,
                     'Share parameters between encoder and decoder.')
flags.DEFINE_integer('latent_dim', 20, 'Size of latent variable.')
# logging
flags.DEFINE_integer('steps_per_checkpoint', 1000,
                     'How many training steps to do per checkpoint.')
flags.DEFINE_integer(
    'print_every', 100,
    'How many steps/minibatches between printing out the loss.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Training directory.')
flags.DEFINE_string('log_dir', './logs', 'Log directory.')
flags.DEFINE_boolean('save', False, 'Save checkpoint files.')

FLAGS = flags.FLAGS

_buckets = [10, 15, 20, 25, 30, 40, 50]


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
    def loop_function(prev, _):
        if output_projection is not None:
            prev = tf.nn.xw_plus_b(prev, output_projection[0],
                                   output_projection[1])
        prev_symbol = tf.argmax(prev, 1)
        emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
        if not update_embedding:
            emb_prev = tf.stop_gradient(emb_prev)
        return emb_prev

    return loop_function


def _revert_inputs(batch_inputs, vocab):
    batch_size = batch_inputs[0].shape[0]
    inputs = [[] for _ in xrange(batch_size)]
    for length_idx in xrange(len(batch_inputs)):
        for batch_idx in xrange(len(batch_inputs[length_idx])):
            inputs[batch_idx].append(batch_inputs[length_idx][batch_idx])
    return inputs


def print_data(batch_encoder_inputs, batch_decoder_inputs, vocab):
    assert len(batch_encoder_inputs) == len(batch_decoder_inputs) - 1
    encoder_inputs = _revert_inputs(batch_encoder_inputs, vocab)
    decoder_inputs = _revert_inputs(batch_decoder_inputs, vocab)

    for enc, dec in zip(encoder_inputs, decoder_inputs):
        print('encoder input > "{}"'.format(map(vocab.token, enc)))
        print('decoder input > "{}"'.format(map(vocab.token, dec)))


class Homotopy(object):
    def __init__(self, num_units, embedding_size, keep_prob, share_param,
                 latent_dim, buckets, vocab):
        self.batch_size = 1
        self.buckets = buckets
        self.global_step = tf.Variable(0, trainable=False)
        self.vocab = vocab
        vocab_size = vocab.size
        self.keep_prob = keep_prob
        self.reg_scale = 0.0

        self.lsent_inputs = []
        self.rsent_inputs = []
        self.decoder_inputs = []

        max_encoder_size, max_decoder_size = buckets[-1], buckets[-1] + 1
        for i in xrange(max_encoder_size):
            self.lsent_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='lsent_encoder{0}'.format(i)))
            self.rsent_inputs.append(tf.placeholder(
                tf.int32,
                shape=[None],
                name='rsent_encoder{0}'.format(i)))
        for i in xrange(max_decoder_size + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.int32, shape=[None],
                name='decoder{0}'.format(i)))

        self.embedding = tf.get_variable('embedding',
                                         [vocab_size, embedding_size],
                                         trainable=True)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units,
                                                 state_is_tuple=True)

        l2_reg = tf.contrib.layers.l2_regularizer(self.reg_scale)

        def homotopy_encoder(encoder_inputs):
            emb_encoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in encoder_inputs]

            with tf.variable_scope('autoencoder', regularizer=l2_reg):
                encoder_scope = 'tied_rnn' if share_param else 'rnn_encoder'
                _, state = tf.nn.rnn(lstm_cell,
                                     emb_encoder_inputs,
                                     dtype=tf.float32,
                                     scope=encoder_scope)

                with tf.variable_scope('latent'):
                    mean = linear(state, latent_dim, True, scope='mean')
                    # log(stddev) or log(var) is all ok for the output
                    log_stddev = linear(state,
                                        latent_dim,
                                        True,
                                        bias_start=-5.0,
                                        scope='log_stddev')
                    stddev = tf.exp(log_stddev)
                    batch_size = tf.shape(state[0])[0]
                    episilon = tf.random_normal([batch_size, latent_dim])
                    z = mean + stddev * episilon
            return z

        def homotopy_decoder(decoder_inputs, initial_state):
            emb_decoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in decoder_inputs]

            with tf.variable_scope('autoencoder', regularizer=l2_reg):
                proj_w = tf.get_variable('proj_w', [num_units, vocab_size])
                proj_b = tf.get_variable('proj_b', [vocab_size])
                # disable when using latent variables
                loop_function = None
                if share_param:
                    tf.get_variable_scope().reuse_variables()
                decoder_scope = 'tied_rnn' if share_param else 'rnn_decoder'
                outputs, _ = tf.nn.seq2seq.rnn_decoder(
                    emb_decoder_inputs,
                    initial_state,
                    lstm_cell,
                    loop_function=loop_function,
                    scope=decoder_scope)
                assert same_shape(outputs[0], (None, num_units))

                logits = [tf.nn.xw_plus_b(output, proj_w, proj_b)
                          for output in outputs]
                assert same_shape(logits[0], (None, vocab_size))
            return logits

        def interpolate(lsent_inputs, rsent_inputs, decoder_inputs):
            z_lsent = homotopy_encoder(lsent_inputs)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                z_rsent = homotopy_encoder(rsent_inputs)
            t = tf.random_uniform([1])[0]
            z = z_lsent * (1 - t) + z_rsent * t
            with tf.variable_scope('autoencoder', regularizer=l2_reg):
                concat = linear(z, 2 * num_units, True, scope='state')
                state = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, concat))
            return homotopy_decoder(decoder_inputs, state)

        self.outputs = []
        for j, seq_length in enumerate(buckets):
            encoder_size, decoder_size = seq_length, seq_length + 1
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if j > 0 else None):
                encoder_size, decoder_size = seq_length, seq_length + 1
                self.outputs.append(
                    interpolate(self.lsent_inputs[:encoder_size],
                                self.rsent_inputs[:encoder_size],
                                self.decoder_inputs[:decoder_size]))
        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, lsent_inputs, rsent_inputs):
        (lsent_encoder_inputs, lsent_decoder_inputs,
         lsent_bucket_id) = lsent_inputs
        (rsent_encoder_inputs, rsent_decoder_inputs,
         rsent_bucket_id) = rsent_inputs
        assert lsent_bucket_id == rsent_bucket_id
        lsent_seq_length = self.buckets[lsent_bucket_id]
        rsent_seq_length = self.buckets[rsent_bucket_id]
        (lsent_encoder_size,
         lsent_decoder_size) = lsent_seq_length, lsent_seq_length + 1
        rsent_encoder_size = rsent_seq_length

        input_feed = {}
        for l in xrange(rsent_encoder_size):
            input_feed[self.rsent_inputs[l].name] = rsent_encoder_inputs[l]
        for l in xrange(lsent_encoder_size):
            input_feed[self.lsent_inputs[l].name] = lsent_encoder_inputs[l]
        for l in xrange(lsent_decoder_size):
            input_feed[self.decoder_inputs[l].name] = lsent_decoder_inputs[l]
        # zero out last target
        input_feed[self.decoder_inputs[lsent_decoder_size].name] = np.zeros(
            [self.batch_size], dtype=np.float32)

        output_feed = []
        for l in xrange(lsent_decoder_size):
            output_feed.append(self.outputs[lsent_bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        return outputs[:]

    def get_batch(self, data, bucket_id):
        seq_length = self.buckets[bucket_id]
        encoder_size, decoder_size = seq_length, seq_length + 1
        encoder_inputs = []
        decoder_inputs = []

        for _ in xrange(self.batch_size):
            encoder_input = random.choice(data[bucket_id])
            decoder_input = encoder_input + [self.vocab.end_index]

            encoder_pad_size = encoder_size - len(encoder_input)
            encoder_inputs.append(encoder_input + [self.vocab.pad_index] *
                                  encoder_pad_size)
            # autoencoder's decoder size == <GO> + encoder
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([self.vocab.start_index] + decoder_input +
                                  [self.vocab.pad_index] * decoder_pad_size)
        assert len(encoder_inputs[0]) == len(decoder_inputs[0]) - 1

        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        for length_idx in xrange(encoder_size):
            batch_encoder_inputs.append(np.array([encoder_inputs[batch_idx][
                length_idx] for batch_idx in xrange(self.batch_size)],
                                                 dtype=np.int32))
        for length_idx in xrange(decoder_size):
            batch_decoder_inputs.append(np.array([decoder_inputs[batch_idx][
                length_idx] for batch_idx in xrange(self.batch_size)],
                                                 dtype=np.int32))
            batch_weight = np.ones([self.batch_size], dtype=np.float32)
            for batch_idx in xrange(self.batch_size):
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if (length_idx == decoder_size - 1 or
                        target == self.vocab.pad_index):
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs, batch_decoder_inputs, batch_weights

    def predict(self, session, lsent, rsent):
        def _inputs_for_bucket(source_ids, max_len):
            assert len(source_ids) < self.buckets[-1]
            eval_set = [[] for _ in self.buckets]
            for bucket_id, bucket_size in enumerate(self.buckets):
                if max_len < bucket_size:
                    eval_set[bucket_id].append(source_ids)
                    break
            encoder_inputs, decoder_inputs, target_weights = self.get_batch(
                eval_set, bucket_id)
            return encoder_inputs, decoder_inputs, bucket_id

        lsent_ids = map(self.vocab.index, lsent.split())
        rsent_ids = map(self.vocab.index, rsent.split())
        max_len = max(len(lsent_ids), len(rsent_ids))
        lsent_inputs = _inputs_for_bucket(lsent_ids, max_len)
        rsent_inputs = _inputs_for_bucket(rsent_ids, max_len)

        outputs = self.step(session, lsent_inputs, rsent_inputs)

        decoder_outputs = []
        for b in xrange(self.batch_size):
            decoder_outputs.append(' '.join([
                self.vocab.token(np.argmax(outputs[l][b]))
                for l in xrange(self.buckets[lsent_inputs[2]] + 1)
            ]))
        return decoder_outputs


def create_model(sess, vocab, forward_only=False):
    model = Homotopy(FLAGS.num_units, FLAGS.embedding_size, FLAGS.keep_prob,
                     FLAGS.share_param, FLAGS.latent_dim, _buckets, vocab)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print('Reading model parameters from {}'.format(
            ckpt.model_checkpoint_path))
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        raise ValueError('No pre-trained model')
    return model


def evaluate():
    vocab = Vocab(os.path.join(FLAGS.data_dir, 'vocab-cased.txt'))

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            model = create_model(sess, vocab, True)
            model.batch_size = 1
            model.keep_prob = 1.0

        while True:
            lsent = raw_input('[1]> ')
            rsent = raw_input('[2]> ')
            if lsent and rsent:
                output = model.predict(sess, lsent, rsent)
                for i, s in enumerate(output):
                    s = s.replace(vocab.unk_token, '?')
                    s = s.replace(vocab.end_token, '').strip()
                    print('[{}]: {}'.format(i + 1, s))


def main(_):
    header('Variational auto-encoder language model')

    header('Configuration')
    print_flags()

    header('Evaluating Homotopy model')
    evaluate()


def header(s):
    print('-' * 50)
    print(s)
    print('-' * 50)


def get_flags():
    try:
        FLAGS.__should_not_exist__
    except:
        pass
    return vars(FLAGS)['__flags']


def print_flags():
    for k, v in get_flags().iteritems():
        print('{:20s}\t= {}'.format(k, v))


def restore_config(config_file):
    with open(config_file) as f:
        flags = pickle.load(f)
    for k, v in flags.iteritems():
        setattr(FLAGS, k, v)


_SAVE_FLAGS = [
    'data_dir',
    'embedding',
    'num_units',
    'embedding_size',
    'latent_dim',
    'use_embedding',
    'learning_rate',
    'max_gradient_norm',
    'batch_size',
    'keep_prob',
    'share_param',
    'log_dir',
]


def save_config(config_file):
    with open(config_file, 'w') as f:
        flags = get_flags()
        saved_flags = {}
        for k in _SAVE_FLAGS:
            saved_flags[k] = flags[k]
        pickle.dump(saved_flags, f)


if __name__ == '__main__':
    tf.app.run()
