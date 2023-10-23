import argparse
import fileinput
import json
from collections import Counter

from schemarec.util import SetEncoder
from jsonlines import jsonlines
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = parser.parse_args()

    objs = dict()
    files = list(args.files if len(args.files) > 0 else ('-',))
    for file in tqdm(files):
        with jsonlines.Reader(fileinput.input(files=(file,))) as reader:
            for v in reader:
                for key, value in v.items():
                    objs[key] = Counter(objs.get(key, dict())) + Counter(value)
    print(json.dumps(objs, cls=SetEncoder))


if __name__ == '__main__':
    main()
