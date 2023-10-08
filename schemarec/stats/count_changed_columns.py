import argparse
import fileinput
import json
from collections import Counter

import jsonlines
from analysis.schemamatching import create_header


def print_stats():
    c = Counter()
    with jsonlines.Reader(fileinput.input(files=args.files if len(args.files) > 0 else ('-',))) as reader:
        for s in reader:
            for obj in s:
                if 'header' not in obj or 'previous-stable' not in obj or 'header' not in obj['previous-stable']:
                    continue

                max_header_length = max(len(obj['header']), len(obj['previous-stable']['header']))
                min_row_count = min(get_row_count(obj), get_row_count(obj['previous-stable']))
                max_row_count = max(get_row_count(obj), get_row_count(obj['previous-stable']))
                max_column_count = max(max(get_row_element_counts(obj)),
                                       max(get_row_element_counts(obj['previous-stable'])))
                max_column_name_length = max(get_max_column_name_length(obj),
                                             get_max_column_name_length(obj['previous-stable']))

                current_header = create_header(obj)
                previous_header = create_header(obj['previous-stable'])
                diff = set(current_header).symmetric_difference(set(previous_header))

                c[(max_header_length, min_row_count, max_row_count, max_column_count, max_column_name_length,
                   len(diff))] += 1

    for k, v in c.items():
        print(json.dumps({
            'max_header_length': k[0],
            'min_row_count': k[1],
            'max_row_count': k[2],
            'max_column_count': k[3],
            'max_column_name_length': k[4],
            'diff': k[5],
            'count': v
        }))


def get_max_column_name_length(obj):
    return max(len(c) for c in obj['header']) if len(obj['header']) > 0 else 0


def get_row_element_counts(obj):
    return (len(c) for c in obj['contentParsed'])


def get_row_count(obj):
    return len(obj['contentParsed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')

    args = parser.parse_args()

    print_stats()
