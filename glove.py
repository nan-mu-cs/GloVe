import tensorflow as tf
import numpy as np

import os
import sys
import threading
import time

flags = tf.app.flags

flags.DEFINE_string("save_path", None, "Directory to write the model and "
                                       "training summaries.")
flags.DEFINE_string("train_data", None, "Training text file. Cooccur_matrix File")
flags.DEFINE_integer("embedding_size", 200, "The embedding dimension size.")
flags.DEFINE_integer(
    "epochs_to_train", 15,
    "Number of epochs to train. Each epoch processes the training data once "
    "completely.")
flags.DEFINE_float("learning_rate", 0.2, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 16,
                     "Number of training examples processed per step "
                     "(size of a minibatch).")
flags.DEFINE_integer("concurrent_steps", 12,
                     "The number of concurrent training steps.")

FLAGS = flags.FLAGS


class GloVe(object):
    def __init__(self, options, session):
        self._save_path = options.save_path
        self._train_data = options.train_data
        self._embedding_size = options.embedding_size
        self._learning_rate = options.learning_rate
        self._batch_size = options.batch_size
        self._concurrent_steps = options.concurrent_steps
        self._learning_rate = options.learning_rate
        self._vocab_size = 0
        self._num_lines = 1
        self._batch_per_epoch = self._num_lines / self._batch_size
        self._x_max = 100
        self._alpha = 0.75
        self._session = session
        self.build_train_graph()
        self.saver = tf.train.Saver()


    def build_train_graph(self):

        target = tf.placeholder(tf.int32,shape=[self._batch_size])
        context = tf.placeholder(tf.int32,shape=[self._batch_size])
        label = tf.placeholder(tf.int32,shape=[self._batch_size,1])
        # emb_w [vocab_size,emb_dim]
        target_emb_w = tf.Variable(
            tf.random_uniform(
                [self._vocab_size, self._embedding_size], -0.5 / self._embedding_size, 0.5 / self._embedding_size),
            name="target_emb_w")

        context_emb_w = tf.Variable(
            tf.random_uniform(
                [self._vocab_size, self._embedding_size], -0.5 / self._embedding_size, 0.5 / self._embedding_size),
            name="context_emb_w")

        target_emb_b = tf.Variable(tf.zeros([self._vocab_size]), name="target_emb_w")
        context_emb_b = tf.Variable(tf.zeros([self._vocab_size]), name="context_emb_w")

        target_w = tf.nn.embedding_lookup(target_emb_w, target)
        context_w = tf.nn.embedding_lookup(context_emb_w, context)

        target_b = tf.nn.embedding_lookup(target_emb_b, target)
        context_b = tf.nn.embedding_lookup(context_emb_b, context)

        diff = tf.matmul(target_w, context_w, transpose_b=True) - target_b - context_b - label
        fdiff = tf.minium(
            diff,
            tf.pow(label / self._x_max, self._alpha) * diff
        )
        loss = diff * fdiff

        if np.isnan(loss):
            print("current loss IS NaN. This should never happen :)")
            sys.exit(1)
        self._loss = loss

        self.global_step = tf.Variable(0, name="global_step")

        self._optimizer = tf.train.AdagradOptimizer(self._learning_rate).minimize(loss, global_step=self.global_step)

    def generate_batch(self, i):
        target = []
        context = []
        label = []
        with open(self._train_data, "r") as f:
            for _ in xrange(0, i * self._batch_size):
                next(f)
            for _ in xrange(i * self._batch_size, (i + 1) * self._batch_size):
                line = f.readline()
                if not line:
                    break
                line = line.split()
                target.append(int(line[0]))
                context.append(int(line[1]))
                label.append(int(line[2]))
        return target, context, label

    def init(self):
        tf.global_variables_initializer().run()
        print('Initialized')

    def run(self):
        average_loss = 0
        for step in xrange(self._batch_per_epoch):
            batch_target,batch_context,batch_label = self.generate_batch(step)
            feed_dict = {target: batch_target,context: batch_context,label:batch_label}
            _, loss_val = self._session.run([self._optimizer, self._loss], feed_dict=feed_dict)
            average_loss += loss_val
            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

def main(_):
    if not FLAGS.save_path or not FLAGS.train_data:
        print("--save_path --train_data must be specified.")
        sys.exit(1)

    with tf.Graph().as_default(), tf.Session() as session:
        model = GloVe(FLAGS, session)
        model.init()
        for epoch in xrange(FLAGS.epochs_to_train):
            model.run()
            if (epoch + 1) % 10 == 0:
                model.saver.save(session,
                                 os.path.join(FLAGS.save_path, "model.ckpt"),
                                 global_step=model.global_step)
        model.saver.save(session,
                         os.path.join(FLAGS.save_path, "model.ckpt"),
                         global_step=model.global_step)

if __name__ == "__main__":
    tf.app.run()
