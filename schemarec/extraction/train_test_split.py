import argparse
import datetime
import fileinput
import random
import uuid
from contextlib import ExitStack
from pathlib import Path

import dateutil.parser as dateparser
import jsonlines

train_last_year_start = datetime.date(2016, 9, 1)
validation_start = datetime.date(2017, 9, 1)
test_start = datetime.date(2018, 9, 1)


def temporal_accept(start_date):
    return lambda version: (start_date is None) or (dateparser.parse(version['validFrom']).date() >= start_date)


def key_accept(keys):
    return lambda version: version['key'] in keys


def temporal_key_accept(keys, start_date):
    a1 = key_accept(keys)
    a2 = temporal_accept(start_date)
    return lambda version: a1(version) and a2(version)


class Split:
    def __init__(self, path, name, name_prefix, criterion, exclusive=True):
        self.name = name
        self.path = Path(path / name)
        self.name_prefix = name_prefix
        self.criterion = criterion
        self.exclusive = exclusive
        self.current_versions = []

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=True)
        self.writer = jsonlines.open(self.path / f'{self.name_prefix}.json', 'w')
        return self.writer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    def accept(self, version):
        return self.criterion is None or self.criterion(version)

    def write(self, obj):
        self.current_versions.append(obj)

    def write_current_versions(self):
        if self.current_versions:
            self.writer.write(self.current_versions)
            self.current_versions = []

    def is_exclusive(self):
        return self.exclusive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='*')
    parser.add_argument('--keys')
    parser.add_argument('--graphranking', action='store_true')
    parser.add_argument('--randomrate', type=float, default=0.1)

    parser.add_argument('--output', default='output')
    args = parser.parse_args()

    path = Path(args.output)

    name_prefix = args.files[0].split('/')[-1].split('.json')[0] if len(args.files) == 1 else uuid.uuid4().hex

    start_date = validation_start if args.graphranking else test_start

    first_split = 'train_ranking' if args.graphranking else 'test'
    second_split = 'train_graph' if args.graphranking else 'train'

    split_groups = []
    if not args.graphranking:
        split_groups.append([
            Split(path, 'temporal/' + first_split, name_prefix, temporal_accept(start_date)),
            Split(path, 'temporal/' + second_split, name_prefix, None),
        ])
    if args.keys is not None:
        keys = set()
        with open(args.keys) as f:
            for line in f:
                keys.add(line.strip())

        if not args.graphranking:
            split_groups.append([
                Split(path, 'spatial/' + first_split, name_prefix, key_accept(keys)),
                Split(path, 'spatial/' + second_split, name_prefix, None),
            ])

        split_groups.append([
            Split(path, 'spatiotemporal/' + first_split, name_prefix, temporal_key_accept(keys, start_date)),
            Split(path, 'spatiotemporal/' + second_split, name_prefix, None),
        ])

    with ExitStack() as stack:
        for splits in split_groups:
            for split in splits:
                stack.enter_context(split)

        with jsonlines.Reader(fileinput.input(files=args.files if len(args.files) > 0 else ('-',))) as reader:
            for table in reader:
                for splits in split_groups:
                    handle_table(splits, table)


def handle_table(splits, table):
    for version in table:
        for split in splits:
            if split.accept(version):
                split.write(version)
                if split.is_exclusive():
                    break
    for split in splits:
        split.write_current_versions()


if __name__ == '__main__':
    main()
