import functools
from collections import Counter

from nltk import word_tokenize


class Table:
    def __init__(self, obj):
        self.obj = obj

    @functools.cached_property
    def header(self):
        return tuple(create_header(self.obj))

    @property
    def rows(self):
        return self.obj['contentParsed'][1:]

    @functools.lru_cache(maxsize=128)
    def column_string(self, column):
        return ' '.join([row[column] for row in self.rows if len(row) > column])

    @functools.lru_cache(maxsize=128)
    def column_tokens(self, column):
        return word_tokenize(self.column_string(column))


def create_header(obj):
    return make_unique(obj['header'], max(len(s) for s in obj['contentParsed']))


def make_unique(header, max_length=0):
    unique_header = Counter()
    result = []
    for h in header:
        result.append((h, unique_header[h]))
        unique_header[h] += 1
    count = 0
    while len(result) < max_length:
        result.append((None, count))
        count += 1
    return result
