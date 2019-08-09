#!/usr/bin/env python3

import csv
import sys

if __name__ == "__main__":
    assert len(sys.argv) == 3
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    with open(input_filename, "r") as input_file:
        reader = csv.reader(input_file)
        rows = list(reader)
        n_samples = len(rows) - 1
        n_features = len(rows[0]) - 2
        labels = sorted(list(set(map(lambda x: x[-1], rows[1:]))))
        n_labels = len(labels)
        with open(output_filename, "w") as output_file:
            print(n_samples, n_features, n_labels, file=output_file)
            print(" ".join(labels), file=output_file)
            for i in rows[1:]:
                print(i[-1], " ".join(i[1:-1]), file=output_file)
