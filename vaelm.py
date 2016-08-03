"""Variational auto-encoder language model"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import time
import sys
import cPickle as pickle
import logging

import numpy as np
import tensorflow as tf

from vocab import Vocab
from model import VariationalAutoEncoder
from datautils import Metadata

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
# model parameters
flags.DEFINE_integer('num_units', 200, 'Size of each LSTM layer.')
flags.DEFINE_boolean('use_embedding', False, 'Use pre-trained embedding')
flags.DEFINE_integer('embedding_size', 350, 'Size of word embedding.')
flags.DEFINE_integer('latent_dim', 10, 'Size of latent variable.')
flags.DEFINE_boolean('share_param', False,
                     'Share parameters between encoder and decoder.')
# hyper-parameters
flags.DEFINE_float('learning_rate', 3e-4, 'Learning rate.')
flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
flags.DEFINE_float('keep_prob', 1.0,
                   'keep probability for dropout regularization.')
flags.DEFINE_integer('batch_size', 64, 'Batch size to use during training.')
flags.DEFINE_float('reg_scale', 0.0, 'Regularization scale.')  # 5e-5
flags.DEFINE_integer('annealing_pivot', 30000, 'Annealing pivot.')
flags.DEFINE_integer('max_steps', 50000,
                     'Number of (global) training steps to perform.')
# logging
flags.DEFINE_integer('steps_per_checkpoint', 1000,
                     'How many training steps to do per checkpoint.')
flags.DEFINE_integer(
    'print_every', 100,
    'How many steps/minibatches between printing out the loss.')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'Training directory.')
flags.DEFINE_string('log_dir', './logs', 'Log directory.')
flags.DEFINE_boolean('save', False, 'Save checkpoint files.')
# other
flags.DEFINE_boolean('eval', False, 'Run a evaluation process.')
flags.DEFINE_boolean('verbose', False, 'Print input data detail.')
flags.DEFINE_boolean('interactive', False, 'Run a interactive shell.')
flags.DEFINE_boolean('word', False, 'Use word or char as input token.')

FLAGS = flags.FLAGS

_buckets = [10, 15, 20, 25, 30, 40, 50]


def read_data(data_path, vocab, max_size=None):
    print('load data from "{}"'.format(data_path))
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(data_path) as source_file:
        counter = 0
        while not max_size or counter < max_size:
            source = source_file.readline().strip()
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
    model = VariationalAutoEncoder(
        FLAGS.learning_rate, FLAGS.batch_size, FLAGS.num_units,
        FLAGS.embedding_size, FLAGS.max_gradient_norm, FLAGS.reg_scale,
        FLAGS.keep_prob, FLAGS.share_param, FLAGS.latent_dim,
        FLAGS.annealing_pivot, _buckets, vocab, forward_only)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
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
        os.path.join(FLAGS.data_dir, 'valid.txt'), vocab,
        FLAGS.max_train_data_size)
    train_set = read_data(
        os.path.join(FLAGS.data_dir, 'train.txt'), vocab,
        FLAGS.max_train_data_size)

    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    metadata = Metadata(FLAGS.log_dir)

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            mtrain = create_model(sess, vocab, False)
            if FLAGS.use_embedding:
                print('Loading word embeddings')
                emb_vecs = np.load(FLAGS.embedding)
                print('Using pre-trained word embedding')
                sess.run(mtrain.embedding.assign(emb_vecs))
        with tf.variable_scope('vaelm', reuse=True):
            mvalid = create_model(sess, vocab, True)

        step_time, loss = 0.0, 0.0
        current_step = 0
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = mtrain.get_batch(
                train_set, bucket_id)
            norm, step_loss, cost_detail = mtrain.step(
                sess, encoder_inputs, decoder_inputs, target_weights,
                bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.print_every
            loss += step_loss / FLAGS.print_every
            current_step += 1

            if current_step % FLAGS.print_every == 0:
                global_step = mtrain.global_step.eval()
                mvalid.global_step.assign(global_step).eval()
                dev_loss = sampled_loss(sess, mvalid, dev_set)
                metadata.add(global_step, 'dev_loss', dev_loss)
                metadata.add(global_step, 'train_loss', loss)
                metadata.add(global_step, 'reconstruction_loss',
                             cost_detail[0])
                metadata.add(global_step, 'kl_loss', cost_detail[1])
                metadata.add(global_step, 'annealing_weight', cost_detail[2])
                ppl = np.exp(loss) if loss < 300 else float('inf')
                print('''global step {} step-time {:.2f} loss {:.2f}'''
                      ''' ppl {:.2f} norm {:.2f}'''
                      .format(global_step, step_time, loss, ppl, norm))
                print('cost detail: {:.2f} {:.2f} {:f}'.format(*cost_detail))
                step_time, loss = 0.0, 0.0

            if current_step % FLAGS.steps_per_checkpoint == 0:
                print('  saving metadata ...')
                metadata.save()

                if FLAGS.save:
                    print('  saving checkpoint ...')
                    sys.stdout.flush()
                    checkpoint_path = os.path.join(FLAGS.checkpoint_dir,
                                                   "model.ckpt")
                    mtrain.saver.save(sess,
                                      checkpoint_path,
                                      global_step=mtrain.global_step)

            if FLAGS.max_steps and current_step % FLAGS.max_steps == 0:
                break


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
        _, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
                                     target_weights, bucket_id, True)
        print('  eval: bucket {} loss {:f}'.format(bucket_id, eval_loss))
        dev_loss += eval_loss
    dev_loss /= nbuckets
    perplexity = np.exp(dev_loss) if dev_loss < 300 else float('inf')
    print('  average sampled dev loss: {:f} ppl {:.2f}'.format(dev_loss,
                                                               perplexity))
    return dev_loss


def evaluate():
    vocab = Vocab(os.path.join(FLAGS.data_dir, 'vocab-cased.txt'))

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            model = create_model(sess, vocab, True)
            model.batch_size = 1

        from djx.nlp.segmenter import segment

        while True:
            source = raw_input('> ')
            if not source:
                continue
            if FLAGS.word:
                source = ' '.join(segment(source)).strip().encode('utf8')
            else:
                source = ' '.join(list(source.decode('utf8'))).encode('utf8')
            output = model.predict(sess, source, FLAGS.verbose)
            output = output.replace(vocab.unk_token, '?')
            print('=> {}'.format(output))


def _start_shell(local_ns=None):
    # An interactive shell is useful for debugging/development.
    import IPython
    user_ns = {}
    if local_ns:
        user_ns.update(local_ns)
    user_ns.update(globals())
    IPython.start_ipython(argv=[], user_ns=user_ns)


def shell():
    vocab = Vocab(os.path.join(FLAGS.data_dir, 'vocab-cased.txt'))

    with tf.Session() as sess:
        with tf.variable_scope('vaelm', reuse=None):
            model = create_model(sess, vocab, True)
            model.batch_size = 1

        from tdb import tvars
        print('[TDB] Load {}'.format(tvars.__name__))
        _start_shell(locals())


def main(_):
    header('Variational auto-encoder language model')

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if FLAGS.save and not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    if FLAGS.save:
        config_file = os.path.join(FLAGS.checkpoint_dir, 'config.pkl')
        if tf.gfile.Exists(config_file):
            restore_config(config_file)
        else:
            save_config(config_file)

    header('Configuration')
    print_flags()

    if FLAGS.eval:
        header('Evaluating model')
        evaluate()
    elif FLAGS.interactive:
        header('Interactive shell')
        shell()
    else:
        header('Training model')
        train()


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
    'reg_scale',
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
