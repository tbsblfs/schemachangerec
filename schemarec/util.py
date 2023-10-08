import json
import logging
import os
from time import perf_counter

from jsonlines import jsonlines
from tqdm import tqdm


def search_files(path, group_changes, progress=False):
    p = tqdm if progress else lambda z: z
    for file in p(os.listdir(path)):
        if file.endswith(".json"):
            with jsonlines.open(os.path.join(path, file)) as reader:
                for obj in reader:
                    group_changes(obj)


def tuplify(listything):
    if isinstance(listything, list):
        return tuple(map(tuplify, listything))
    if isinstance(listything, dict):
        return {k: tuplify(v) for k, v in listything.items()}
    return listything


class SetEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class catchtime_local:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.time = perf_counter()
        logging.info(f'Starting {self.name}')
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.time
        logging.info(f'Finished {self.name} in: {self.time:.3f} seconds')

class TableFilter:
    def __init__(self, min_row_count, max_column_count, max_column_name_length):
        self.min_row_count = min_row_count
        self.max_column_count = max_column_count
        self.max_column_name_length = max_column_name_length

    def matches(self, table):
        if len(table.rows) < self.min_row_count:
            return False

        if len(table.header) > self.max_column_count:
            return False

        if any(len(c[0]) > self.max_column_name_length for c in table.header if c[0]):
            return False
        return True

def fulfills_criteria(args, current_table):
    if len(current_table.rows) < args.min_row_count:
        return False

    current_header = current_table.header
    if len(current_header) > args.max_column_count:
        return False

    if any(len(c[0]) > args.max_column_name_length for c in current_header if c[0]):
        return False
    return True
