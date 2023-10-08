import argparse
import fileinput
import json
import random
from collections import defaultdict

import jsonlines
from analysis.schemamatching.schema_matching import make_unique
from tqdm import tqdm


class ReservoirSample:
    def __init__(self, n):
        self.n = n
        self.reservoir = []
        self.count = 0

    def add(self, obj):
        self.count += 1
        if len(self.reservoir) < self.n:
            self.reservoir.append(obj)
        else:
            m = random.randint(0, self.count)
            if m < self.n:
                self.reservoir[m] = obj

    def get_samples(self):
        return self.reservoir


def get_samples():
    reservoirs = defaultdict(lambda: ReservoirSample(args.n))

    # show progress
    files = list(args.files if len(args.files) > 0 else ('-',))
    for file in tqdm(files):
        with jsonlines.Reader(fileinput.input(files=(file,))) as reader:
            for s in reader:
                for obj in s:
                    if 'header' not in obj or 'previous-stable' not in obj or 'header' not in obj['previous-stable']:
                        continue

                    current_header = set(make_unique(obj['header']))
                    previous_header = set(make_unique(obj['previous-stable']['header']))
                    diff = current_header.symmetric_difference(previous_header)
                    group = int(len(diff)).bit_length()
                    reservoirs[group].add(obj)

    return {k: v.get_samples() for k, v in reservoirs.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    parser.add_argument('--n', type=int, default=30, help='number of samples per group')

    args = parser.parse_args()

    print(json.dumps(get_samples()))
