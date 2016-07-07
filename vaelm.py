"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import random
import time
import sys

import numpy as np
import tensorflow as tf

from vocab import Vocab

flags = tf.flags

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
flags.DEFINE_integer('batch_size', 25, 'Batch size to use during training.')
flags.DEFINE_integer('embedding_size', 300, 'Size of word embedding.')
flags.DEFINE_integer('num_units', 150, 'Size of each LSTM layer.')
flags.DEFINE_integer('num_layers', 1, 'Number of layers in the model.')
flags.DEFINE_integer('num_classes', 5, 'Number of similarity rating classes.')
flags.DEFINE_string('data_dir', './data', 'Data directory')
flags.DEFINE_string('train_dir', './checkpoints', 'Training directory.')
flags.DEFINE_string('log_dir', './logs', 'Log directory.')
flags.DEFINE_integer('max_train_data_size', 0,
                     'Limit on the size of training data (0: no limit).')
flags.DEFINE_integer('steps_per_checkpoint', 200,
                     'How many training steps to do per checkpoint.')
flags.DEFINE_boolean('save', False, 'save checkpoint files.')

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


class VariationalAutoEncoder(object):
    def __init__(self, learning_rate, batch_size, num_units, embedding_size,
                 max_gradient_norm, buckets, vocab, forward_only):
        self.batch_size = batch_size
        self.buckets = buckets
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.vocab = vocab
        vocab_size = vocab.size
        self.reg_scale = 1e-4

        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        max_encoder_size, max_decoder_size = buckets[-1], buckets[-1] + 1
        for i in xrange(max_encoder_size):
            self.encoder_inputs.append(tf.placeholder(
                tf.int32, shape=[None],
                name='encoder{0}'.format(i)))
        for i in xrange(max_decoder_size + 1):
            self.decoder_inputs.append(tf.placeholder(
                tf.int32, shape=[None],
                name='decoder{0}'.format(i)))
            self.target_weights.append(tf.placeholder(tf.float32,
                                                      shape=[None],
                                                      name='weight{0}'.format(
                                                          i)))
        self.targets = [self.decoder_inputs[i + 1]
                        for i in xrange(len(self.decoder_inputs) - 1)]

        self.embedding = tf.get_variable('embedding',
                                         [vocab_size, embedding_size],
                                         trainable=False)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units,
                                                 state_is_tuple=True)

        def autoencoder(encoder_inputs, decoder_inputs, targets, weights):
            emb_encoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in encoder_inputs]
            emb_decoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in decoder_inputs]

            l2_reg = tf.contrib.layers.l2_regularizer(self.reg_scale)
            with tf.variable_scope('autoencoder', regularizer=l2_reg):
                _, state = tf.nn.rnn(lstm_cell,
                                     emb_encoder_inputs,
                                     dtype=tf.float32,
                                     scope='encoder')
                if not forward_only:
                    outputs, _ = tf.nn.seq2seq.rnn_decoder(emb_decoder_inputs,
                                                           state,
                                                           lstm_cell,
                                                           scope='decoder')
                else:
                    loop_function = _extract_argmax_and_embed(self.embedding)
                    outputs, _ = tf.nn.seq2seq.rnn_decoder(
                        emb_decoder_inputs,
                        state,
                        lstm_cell,
                        loop_function=loop_function,
                        scope='decoder')

            loss = tf.nn.seq2seq.sequence_loss(outputs, targets, weights)
            regularizers = \
                tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss += regularizers
            return outputs, loss

        self.losses = []
        self.outputs = []
        for j, seq_length in enumerate(buckets):
            encoder_size, decoder_size = seq_length, seq_length + 1
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if j > 0 else None):
                bucket_outputs, loss = autoencoder(
                    self.encoder_inputs[:encoder_size],
                    self.decoder_inputs[:decoder_size],
                    self.targets[:decoder_size],
                    self.target_weights[:decoder_size])
                self.outputs.append(bucket_outputs)
                self.losses.append(loss)

        params = tf.trainable_variables()
        self.updates = []
        self.gradient_norms = []
        if not forward_only:
            opt = tf.train.AdamOptimizer(learning_rate)
            for b in xrange(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(
                    gradients, max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        seq_length = self.buckets[bucket_id]
        encoder_size, decoder_size = seq_length, seq_length + 1

        input_feed = {}
        for l in xrange(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        # zero out last target
        input_feed[self.decoder_inputs[decoder_size].name] = np.zeros(
            [self.batch_size], dtype=np.float32)

        if forward_only:
            loss = session.run(self.losses[bucket_id], input_feed)
        else:
            _, loss = session.run([
                self.updates[bucket_id],
                self.losses[bucket_id],
            ], input_feed)
        return loss

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
            # autoencoder's decoder size == encoder + <EOS>
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


def create_model(sess, vocab, embedding, forward_only=False):
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
    if embedding is not None:
        print('Using pre-trained word embedding')
        sess.run(model.embedding.assign(embedding))
    return model


def train():
    data_dir = 'data/sick/'
    vocab = Vocab(data_dir + 'vocab-cased.txt')
    print('Loading word embeddings')
    # use only vectors in vocabulary (not necessary, but gives faster training)
    emb_vecs = np.load(data_dir + 'sick.300d.npy')

    print('Reading development and training data (limit: {}).'.format(
        FLAGS.max_train_data_size))
    dev_set = read_data(data_dir + 'dev/', vocab, FLAGS.max_train_data_size)
    train_set = read_data(data_dir + 'train/', vocab,
                          FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    metadata = Metadata(FLAGS.log_dir)

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            model = create_model(sess, vocab, emb_vecs, False)

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
            step_loss = model.step(sess, encoder_inputs, decoder_inputs,
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
        eval_loss = model.step(session, encoder_inputs, decoder_inputs,
                               target_weights, bucket_id, True)
        print('  eval: bucket {} loss {:f}'.format(bucket_id, eval_loss))
        dev_loss += eval_loss
    dev_loss /= nbuckets
    print('  average sampled dev loss: {:f}'.format(dev_loss))
    return dev_loss


def evaluate():
    data_dir = 'data/sick/'
    vocab = Vocab(data_dir + 'vocab-cased.txt')
    print('Loading word embeddings')
    emb_vecs = np.load(data_dir + 'sick.300d.npy')

    with tf.Session() as sess:
        with tf.variable_scope('model', reuse=None):
            model = create_model(sess, vocab, emb_vecs, True)

        while True:
            print('input first sentence')
            source = raw_input('> ')
            print('input second sentence')
            target = raw_input('> ')
            output = model.predict(sess, source, target)
            print('similary evaluation result ([1-5]): {:.2f}'.format(output))


def main(_):
    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
