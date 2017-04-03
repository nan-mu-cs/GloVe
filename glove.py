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
flags.DEFINE_integer("vocab_size", None, "The vocab size.")
flags.DEFINE_integer("matrix_size", None, "Matrix Size.")

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
        self._vocab_size = options.vocab_size
        self._num_lines = options.matrix_size
        self._batch_per_epoch = self._num_lines / self._batch_size
        self._x_max = 100
        self._alpha = 0.75
        self._session = session
        self.build_train_graph()
        self.saver = tf.train.Saver()

    def build_train_graph(self):
        self.target, self.context, self.label = self.read_data()
        # self.target = tf.placeholder(tf.int32, shape=[self._batch_size], name="target")
        # self.context = tf.placeholder(tf.int32, shape=[self._batch_size], name="context")
        # self.label = tf.placeholder(tf.float32, shape=[self._batch_size], name="label")
        alpha = tf.constant(self._alpha, dtype=tf.float32)
        x_max = tf.constant(self._x_max, dtype=tf.float32)

        # target_emb_w [vocab_size,emb_dim]
        target_emb_w = tf.Variable(
            tf.random_uniform(
                [self._vocab_size, self._embedding_size], -0.5 / self._embedding_size, 0.5 / self._embedding_size),
            name="target_emb_w")

        # context_emb_w [vocab_size,emb_dim]
        context_emb_w = tf.Variable(
            tf.random_uniform(
                [self._vocab_size, self._embedding_size], -0.5 / self._embedding_size, 0.5 / self._embedding_size),
            name="context_emb_w")

        # target_emb_b [vocab_size]
        target_emb_b = tf.Variable(tf.zeros([self._vocab_size]), name="target_emb_w")
        # context_emb_b [vocab_size]
        context_emb_b = tf.Variable(tf.zeros([self._vocab_size]), name="context_emb_w")

        # target_w [batch_size,emb_size]
        target_w = tf.nn.embedding_lookup(target_emb_w, self.target)
        # context_w [batch_size,emb_size]
        context_w = tf.nn.embedding_lookup(context_emb_w, self.context)

        # target_b [batch_size]
        target_b = tf.nn.embedding_lookup(target_emb_b, self.target)
        # context_b [bath_size]
        context_b = tf.nn.embedding_lookup(context_emb_b, self.context)

        diff = tf.square(
            tf.reduce_sum(tf.multiply(target_w, context_w), axis=1) - target_b - context_b - tf.log(self.label)
        )

        fdiff = tf.minimum(
            diff,
            tf.pow(self.label / x_max, alpha) * diff
        )

        loss = tf.reduce_mean(tf.multiply(diff, fdiff))

        self._loss = loss

        self.global_step = tf.Variable(0, name="global_step")

        self._optimizer = tf.train.AdagradOptimizer(self._learning_rate).minimize(loss, global_step=self.global_step)

    def generate_batch(self, i):
        target = np.ndarray(shape=self._batch_size, dtype=np.int32)
        context = np.ndarray(shape=self._batch_size, dtype=np.int32)
        label = np.ndarray(shape=self._batch_size, dtype=np.int32)
        with open(self._train_data, "r") as f:
            for _ in xrange(0, i * self._batch_size):
                f.readline()
            line_index = 0
            for _ in xrange(i * self._batch_size, (i + 1) * self._batch_size):
                line = f.readline()

                if not line:
                    break
                line = line.split()
                target[line_index] = int(line[0])
                context[line_index] = int(line[1])
                label[line_index] = int(line[2])
                line_index += 1
        return target, context, label

    def read_data(self):
        filename_queue = tf.train.string_input_producer([self._train_data])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'target': tf.FixedLenFeature([], tf.int64),
                'context': tf.FixedLenFeature([], tf.int64),
                'label': tf.FixedLenFeature([], tf.int64),
            })

        target = features['target']
        context = features['context']
        label = data['label']
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * self._batch_size
        target_batch, context_batch, label_batch = tf.train.shuffle_batch(
            [target, context, label], batch_size=self._batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue)
        return target_batch, context_batch, label_batch

    def init(self):
        #self.target, self.context, self.label = self.read_data()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        self._session.run(init_op)
        print('Initialized')

    def run(self):
        average_loss = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in xrange(self._batch_per_epoch):
            #batch_target, batch_context, batch_label = self.read_data()
            #feed_dict = {self.target: batch_target, self.context: batch_context, self.label: batch_label}
            _, loss_val = self._session.run(
                [self._optimizer, self._loss])
            if np.isnan(loss_val):
                print("current loss IS NaN. This should never happen :)")
                sys.exit(1)

            average_loss += loss_val
            if step % 200 == 0:
                if step > 0:
                    average_loss /= 200
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Step: %d Avg_loss: %f' % (step, average_loss))
                average_loss = 0
        coord.request_stop()
        coord.join(threads)


def main(_):
    if not FLAGS.save_path or not FLAGS.train_data:
        print("--save_path --train_data must be specified.")
        sys.exit(1)

    with tf.Graph().as_default(), tf.Session() as session:
        model = GloVe(FLAGS, session)
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
