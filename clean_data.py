import tensorflow as tf
import collections
import os
import sys
import json

flags = tf.app.flag

flags.DEFINE_string("save_path", None, "Directory to write cleaned data")
flags.DEFINE_string("data_file", None, "File of data to be cleaned")
flags.DEFINE_integer("window_size", 5,
                     "The number of words to predict to the left and right "
                     "of the target word.")
lags.DEFINE_integer("min_count", 5,
                    "The minimum number of word occurrences for it to be "
                    "included in the vocabulary.")
flags.DEFINE_float("subsample", 1e-3,
                   "Subsample threshold for word occurrence. Words that appear "
                   "with higher frequency will be randomly down-sampled. Set "
                   "to 0 to disable.")


class CleanData(object):
    def __init__(self, options):
        self._save_path = options.save_path
        self._data_file = options.data_file
        self._min_count = options.min_count
        self._subsample = options.subsample
        self.dictionary = dict()
        self.reverse_dictionary = dict()

        if not self._data_file or not self._save_path:
            print("--save_path and --data_file must be specified.")
            sys.exit(1)

    def read_data(self):
        with open(self._data_file) as f:
            data = f.read().split()
            print("data file contains %d words" % len(data))
        return data

    def build_dataset(self):
        data = collections.Counter(self.read_data()).most_common()
        endIndex = 0
        for index, item in reversed(data):
            if item[1] >= self._min_count:
                endIndex = index
                break
        data = data[0:endIndex]

        for word, _ in data:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        with open(os.path.join(self._save_path, "vocab.txt"), "w") as f:
            f.write(json.dumps(data))
            print("Write vocab into %s." % os.path.join(self._save_path, "vocab.txt"))
        return data


if __name__ == "__main__":
    clean_data = CleanData(flags.FLAGS)
