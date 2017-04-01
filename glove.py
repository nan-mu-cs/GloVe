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
        self._vocab_size = 0
        self._session = session

    def forward(self,target,context):

        #emb [vocab_size,emb_dim]
        emb = tf.Variable(
            tf.random_uniform(
                [self._vocab_size, self._embedding_size], -0.5/self._embedding_size, 0.5/self._embedding_size),
            name="emb")

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


def main(_):
    if not FLAGS.save_path or not FLAGS.train_data:
        print("--save_path --train_data must be specified.")
        sys.exit(1)

    with tf.Graph().as_default(), tf.Session() as session:
        model = GloVe(FLAGS, session)


if __name__ == "__main__":
    tf.app.run()
