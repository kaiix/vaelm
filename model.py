"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import random

import numpy as np
import tensorflow as tf

from nnutils import same_shape


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


def print_data(batch_encoder_inputs, batch_decoder_inputs,
               batch_target_weights, vocab):
    assert len(batch_encoder_inputs) == len(batch_decoder_inputs) - 1
    assert len(batch_target_weights) == len(batch_decoder_inputs)
    encoder_inputs = _revert_inputs(batch_encoder_inputs, vocab)
    decoder_inputs = _revert_inputs(batch_decoder_inputs, vocab)
    target_weights = _revert_inputs(batch_target_weights, vocab)

    for enc, dec, w in zip(encoder_inputs, decoder_inputs, target_weights):
        print('encoder input > "{}"'.format(map(vocab.token, enc)))
        print('decoder input > "{}"'.format(map(vocab.token, dec)))
        print('target weights > "{}"'.format(list(zip(
            map(vocab.token, dec[1:]), w))))


class VariationalAutoEncoder(object):
    def __init__(self, learning_rate, batch_size, num_units, embedding_size,
                 max_gradient_norm, reg_scale, buckets, vocab, forward_only):
        self.batch_size = batch_size
        self.buckets = buckets
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.vocab = vocab
        vocab_size = vocab.size
        self.reg_scale = reg_scale

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
                                         trainable=True)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units,
                                                 state_is_tuple=True)

        def autoencoder(encoder_inputs, decoder_inputs, targets, weights):
            emb_encoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in encoder_inputs]
            emb_decoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in decoder_inputs]
            assert len(emb_encoder_inputs) == len(emb_decoder_inputs) - 1
            assert len(targets) == len(weights)

            l2_reg = tf.contrib.layers.l2_regularizer(self.reg_scale)
            with tf.variable_scope('autoencoder', regularizer=l2_reg):
                _, state = tf.nn.rnn(lstm_cell,
                                     emb_encoder_inputs,
                                     dtype=tf.float32,
                                     scope='encoder')

                proj_w = tf.get_variable('proj_w', [num_units, vocab_size])
                proj_b = tf.get_variable('proj_b', [vocab_size])
                if forward_only:
                    loop_function = _extract_argmax_and_embed(self.embedding,
                                                              (proj_w, proj_b))
                else:
                    loop_function = None

                outputs, _ = tf.nn.seq2seq.rnn_decoder(
                    emb_decoder_inputs,
                    state,
                    lstm_cell,
                    loop_function=loop_function,
                    scope='decoder')
                assert same_shape(outputs[0], (None, num_units))

                logits = [tf.nn.xw_plus_b(output, proj_w, proj_b)
                          for output in outputs]
                assert same_shape(logits[0], (None, vocab_size))

            loss = tf.nn.seq2seq.sequence_loss(logits, targets, weights)
            regularizers = \
                tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            loss += regularizers
            return logits, loss

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

        if not forward_only:
            output_feed = [
                self.updates[bucket_id],
                self.losses[bucket_id],
            ]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], None  # loss
        else:
            return outputs[0], outputs[1:]  # loss, logits

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

    def predict(self, session, sentence):
        source_ids = map(self.vocab.index, sentence.split())
        assert len(source_ids) < self.buckets[-1]
        eval_set = [[] for _ in self.buckets]
        for bucket_id, bucket_size in enumerate(self.buckets):
            if len(source_ids) < bucket_size:
                eval_set[bucket_id].append(source_ids)
                break
        encoder_inputs, decoder_inputs, target_weights = self.get_batch(
            eval_set, bucket_id)
        # print_data(encoder_inputs, decoder_inputs, target_weights, self.vocab)
        _, outputs = self.step(session, encoder_inputs, decoder_inputs,
                               target_weights, bucket_id, True)
        assert len(outputs) == self.buckets[bucket_id] + 1
        assert same_shape(outputs[0], (self.batch_size, self.vocab.size))
        decoder_outputs = []
        for b in xrange(self.batch_size):
            decoder_outputs.append(' '.join([
                self.vocab.token(np.argmax(outputs[l][b]))
                for l in xrange(self.buckets[bucket_id] + 1)
            ]))
        return decoder_outputs
