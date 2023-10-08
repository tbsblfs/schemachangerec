import argparse
import fileinput
import json
from collections import Counter

import jsonlines
from tqdm import tqdm

from schemarec.util import SetEncoder, tuplify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    parser.add_argument('--nodes', action='store_true', help='merge nodes')
    args = parser.parse_args()

    objs = dict()
    files = list(args.files if len(args.files) > 0 else ('-',))
    for file in tqdm(files):
        with jsonlines.Reader(fileinput.input(files=(file,))) as reader:
            for v in reader:
                key = tuplify(v[0]) if args.nodes else tuplify(v[0:2])

                if key in objs:
                    objs[key] = reducer(objs[key], v, 1 if args.nodes else 2)
                else:
                    objs[key] = v
    for n in objs.values():
        print(json.dumps(n, cls=SetEncoder))


def reducer(accumulator, element, idx):
    for key, value in element[idx].items():
        if isinstance(value, int):
            accumulator[idx][key] = accumulator[idx].get(key, 0) + value
        elif isinstance(value, list):
            accumulator[idx][key] = set(tuplify(accumulator[idx].get(key, set()))) | set(tuplify(value))
        elif isinstance(value, dict):
            accumulator[idx][key] = Counter(accumulator[idx].get(key, dict())) + Counter(value)
        else:
            raise Exception("Unknown type: {}".format(type(value)))
    return accumulator


if __name__ == '__main__':
    main()
