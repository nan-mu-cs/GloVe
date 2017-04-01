import numpy as np
import collections
import argparse
import os
import sys
import json

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", type=str, default=None, help="Directory to write cleaned data")
parser.add_argument("--data_file", type=str, default=None, help="File of data to be cleaned")
parser.add_argument("--window_size", type=int, default=5,
                    help="The number of words to predict to the left and right "
                         "of the target word.")
parser.add_argument("--min_count", type=int, default=5,
                    help="The minimum number of word occurrences for it to be "
                         "included in the vocabulary.")
parser.add_argument("--subsample", type=float, default=1e-3,
                    help="Subsample threshold for word occurrence. Words that appear "
                         "with higher frequency will be randomly down-sampled. Set "
                         "to 0 to disable.")
args = parser.parse_args()
print args


class CleanData(object):
    def __init__(self, options):
        self._save_path = options.save_path
        self._data_file = options.data_file
        self._min_count = options.min_count
        self._subsample = options.subsample
        self._vocab_size = 0
        self._window_size = options.window_size
        self.dictionary = dict()
        self.reverse_dictionary = dict()

        if not self._data_file or not self._save_path:
            print("--save_path and --data_file must be specified.")
            sys.exit(1)

    def read_data(self):
        with open(self._data_file, "r") as f:
            data = f.read().split()
            print("data file contains %d words" % len(data))
        return data

    def build_dataset(self):
        data = collections.Counter(self.read_data()).most_common()
        end_index = 0
        for index, item in enumerate(reversed(data)):
            if item[1] >= self._min_count:
                end_index = index
                break
        data = data[0:-end_index]
        self._vocab_size = len(data)
        print("vocab size is %d." % self._vocab_size)

        for word, _ in data:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        with open(os.path.join(self._save_path, "vocab.txt"), "w") as f:
            f.write(json.dumps(data))
            print("Write vocab into %s." % os.path.join(self._save_path, "vocab.txt"))
        return data

    def update_coocur_line(self, line):
        window = self._window_size
        result = []
        line = line.split()
        for index, word in enumerate(line):
            if word in self.dictionary:
                target_index = self.dictionary[word]
            else:
                continue
            for i in range(max(0, index - window), min(len(line), index + window)):
                if line[i] in self.dictionary:
                    context_index = self.dictionary[line[i]]
                    result.append((target_index, context_index))
                else:
                    continue
        return result

    def build_cooccur(self):
        line_number = 0
        cooccur_matrix = dict()
        with open(self._data_file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = self.update_coocur_line(line)
                for target_word, context_word in line:
                    key = str(target_word) + "-" + str(context_word)
                    if key in cooccur_matrix:
                        cooccur_matrix[key] += 1
                    else:
                        cooccur_matrix[key] = 1
                if (line_number + 1) % 1000 == 0:
                    print("Processed %d lines" % (line_number + 1))
                line_number += 1

        print("Finish processed files")
        with open(os.path.join(self._save_path, "cooccur_matrix.txt"), "w") as output:
            for i in range(0, self._vocab_size):
                for j in range(0, self._vocab_size):
                    key = str(i) + "-" + str(j)
                    if key in cooccur_matrix:
                        f.write("%d %d %d\n" % (i, j, cooccur_matrix[key]))

            print("Save cooccur matrix into %s" % (os.path.join(self._save_path, "cooccur_matrix.txt")))

    def clean(self):
        self.build_dataset()
        self.build_cooccur()


if __name__ == "__main__":
    clean_data = CleanData(args)
    clean_data.clean()
