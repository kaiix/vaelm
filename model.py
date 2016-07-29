"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import random

import numpy as np
import tensorflow as tf

from nnutils import same_shape
from nnutils import linear


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
                 max_gradient_norm, reg_scale, keep_prob, share_param,
                 latent_dim, lr_decay, annealing_pivot, buckets, vocab,
                 forward_only):
        self.batch_size = batch_size
        self.buckets = buckets
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.vocab = vocab
        vocab_size = vocab.size
        self.reg_scale = reg_scale
        self.forward_only = forward_only
        self.keep_prob = keep_prob

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

        def annealing_schedule(t, pivot):
            return tf.nn.sigmoid((t - pivot) / pivot * 100)

        def unk_dropout(x, keep_prob, unk_index):
            # TODO: don't dropout <GO>
            with tf.op_scope([x], None, 'dropout'):
                x = tf.convert_to_tensor(x, name='x')
                if isinstance(keep_prob, float) and not 0 < keep_prob <= 1:
                    raise ValueError(
                        "keep_prob must be a scalar tensor or a float in the "
                        "range (0, 1], got %g" % keep_prob)
                keep_prob = tf.convert_to_tensor(keep_prob,
                                                 dtype=tf.float32,
                                                 name="keep_prob")

                # uniform [keep_prob, 1.0 + keep_prob)
                random_tensor = keep_prob
                random_tensor += tf.random_uniform(tf.shape(x))
                # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
                binary_tensor = tf.floor(random_tensor)
                ret = tf.select(
                    tf.greater(binary_tensor, 0), x, tf.fill(
                        tf.shape(x), unk_index))
                ret.set_shape(x.get_shape())
                return ret

        def autoencoder(encoder_inputs, decoder_inputs, targets, weights):
            emb_encoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in encoder_inputs]
            decoder_inputs = [
                unk_dropout(i, self.keep_prob, self.vocab.unk_index)
                for i in decoder_inputs
            ]
            emb_decoder_inputs = [tf.nn.embedding_lookup(self.embedding, i)
                                  for i in decoder_inputs]
            assert len(emb_encoder_inputs) == len(emb_decoder_inputs) - 1
            assert len(targets) == len(weights)

            l2_reg = tf.contrib.layers.l2_regularizer(self.reg_scale)
            with tf.variable_scope('autoencoder', regularizer=l2_reg):
                encoder_scope = 'tied_rnn' if share_param else 'rnn_encoder'
                _, state = tf.nn.rnn(lstm_cell,
                                     emb_encoder_inputs,
                                     dtype=tf.float32,
                                     scope=encoder_scope)

                proj_w = tf.get_variable('proj_w', [num_units, vocab_size])
                proj_b = tf.get_variable('proj_b', [vocab_size])
                if forward_only:
                    loop_function = _extract_argmax_and_embed(self.embedding,
                                                              (proj_w, proj_b))
                else:
                    loop_function = None
                # disable when using latent variables
                loop_function = None

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

                concat = linear(z, 2 * num_units, True, scope='state')
                state = tf.nn.rnn_cell.LSTMStateTuple(*tf.split(1, 2, concat))

                if share_param:
                    tf.get_variable_scope().reuse_variables()
                decoder_scope = 'tied_rnn' if share_param else 'rnn_decoder'
                outputs, _ = tf.nn.seq2seq.rnn_decoder(
                    emb_decoder_inputs,
                    state,
                    lstm_cell,
                    loop_function=loop_function,
                    scope=decoder_scope)
                assert same_shape(outputs[0], (None, num_units))

                logits = [tf.nn.xw_plus_b(output, proj_w, proj_b)
                          for output in outputs]
                assert same_shape(logits[0], (None, vocab_size))

            # cross entropy loss = -sum(y * log(p(y))
            reconstruction_loss = tf.nn.seq2seq.sequence_loss(logits, targets,
                                                              weights)
            kl_loss = -0.5 * (
                1.0 + 2 * log_stddev - tf.square(mean) - tf.square(stddev))
            kl_loss = tf.reduce_sum(kl_loss) / tf.cast(batch_size, tf.float32)

            annealing_weight = annealing_schedule(
                tf.cast(self.global_step, tf.float32), annealing_pivot)
            # loss = -E[log(p(x))] + D[q(z)||p(z)]
            loss = reconstruction_loss + annealing_weight * kl_loss
            if reg_scale > 0.0:
                regularizers = tf.add_n(tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES))
                loss += regularizers
            return logits, loss, (reconstruction_loss, kl_loss,
                                  annealing_weight)

        self.losses = []
        self.outputs = []
        self.costs = []
        for j, seq_length in enumerate(buckets):
            encoder_size, decoder_size = seq_length, seq_length + 1
            with tf.variable_scope(tf.get_variable_scope(),
                                   reuse=True if j > 0 else None):
                bucket_outputs, loss, cost_detail = autoencoder(
                    self.encoder_inputs[:encoder_size],
                    self.decoder_inputs[:decoder_size],
                    self.targets[:decoder_size],
                    self.target_weights[:decoder_size])
                self.outputs.append(bucket_outputs)
                self.losses.append(loss)
                self.costs.append(cost_detail)

        self.updates = []
        self.gradient_norms = []
        if not forward_only:
            if lr_decay > 0.0:
                self.learning_rate = tf.train.exponential_decay(
                    learning_rate,
                    self.global_step,
                    1000,
                    lr_decay,
                    staircase=True)
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                print('  training with SGD optimizer')
            else:
                opt = tf.train.AdamOptimizer(self.learning_rate)
                print('  training with Adam optimizer')
            params = tf.trainable_variables()
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
                self.gradient_norms[bucket_id],
                self.losses[bucket_id],
                self.costs[bucket_id][0],
                self.costs[bucket_id][1],
                self.costs[bucket_id][2],
            ]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in xrange(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return (outputs[1], outputs[2],
                    outputs[3:])  # gradient norm, loss, (xent, -kl, annealing)
        else:
            return None, outputs[0], outputs[1:]  # gradient norm, loss, logits

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

    def predict(self, session, sentence, verbose=False):
        source_ids = map(self.vocab.index, sentence.split())
        assert len(source_ids) < self.buckets[-1]
        eval_set = [[] for _ in self.buckets]
        for bucket_id, bucket_size in enumerate(self.buckets):
            if len(source_ids) < bucket_size:
                eval_set[bucket_id].append(source_ids)
                break
        encoder_inputs, decoder_inputs, target_weights = self.get_batch(
            eval_set, bucket_id)
        if verbose:
            print_data(encoder_inputs, decoder_inputs, target_weights,
                       self.vocab)
        _, _, outputs = self.step(session, encoder_inputs, decoder_inputs,
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
