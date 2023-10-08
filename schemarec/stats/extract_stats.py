import argparse
import fileinput
import json
import uuid
from collections import Counter
from pathlib import Path

import tqdm as tqdm
from analysis.schemamatching import Table, create_correspondences, create_config
from analysis.util.util import SetEncoder
from jsonlines import jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    parser.add_argument('--min-row-count', type=int, help='minimum row count', default=1)
    parser.add_argument('--max-column-count', type=int, help='maximum column count', default=100)
    parser.add_argument('--max-column-name-length', type=int, help='maximum column name length', default=150)
    parser.add_argument('--output', help='output folder', type=str, required=True)

    args = parser.parse_args()

    config = create_config()

    users = Counter()
    change_types = Counter()
    change_counts = Counter()
    change_sizes = Counter()
    diff_types = Counter()

    with jsonlines.Reader(fileinput.input(files=args.files if len(args.files) > 0 else ('-',))) as reader:
        for v in tqdm.tqdm(reader):
            if len(v) == 0:
                change_counts[0] += 1
                continue

            valid_changes = 0
            for s in v:
                if 'header' not in s:
                    change_types['no-header'] += 1
                    continue

                if 'previous-stable' not in s or 'header' not in s['previous-stable']:
                    change_types['no-previous-header'] += 1
                    continue

                current_table = Table(s)
                if not fulfills_criteria(args, current_table):
                    change_types['not-fulfilling-criteria'] += 1
                    continue

                prev_table = Table(s['previous-stable'])
                if not fulfills_criteria(args, prev_table):
                    change_types['not-fulfilling-criteria-previous'] += 1
                    continue

                change_types['fulfilling-criteria'] += 1

                sym_diff_size = len(set(current_table.header).symmetric_difference(prev_table.header))
                change_sizes[sym_diff_size] += 1

                corresponding_columns = create_correspondences(s, config)
                renaming_count = len([c for c in corresponding_columns if c[0] != c[1]])
                added_count = len(set(current_table.header) - set(prev_table.header)) - renaming_count
                removed_count = len(set(prev_table.header) - set(current_table.header)) - renaming_count

                if renaming_count > 0 and added_count == 0 and removed_count == 0:
                    diff_types['renaming'] += 1
                elif renaming_count == 0 and added_count > 0 and removed_count == 0:
                    diff_types['adding'] += 1
                elif renaming_count == 0 and added_count == 0 and removed_count > 0:
                    diff_types['removing'] += 1
                elif renaming_count > 0 or added_count > 0 or removed_count > 0:
                    diff_types['other'] += 1
                else:
                    diff_types['no-change'] += 1

                valid_changes += 1

                if 'id' in s['user']:
                    users[s['user']['id']] += 1

            change_counts[valid_changes] += 1

    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)

    name_prefix = args.files[0].split('/')[-1].split('.json')[0] if len(args.files) == 1 else uuid.uuid4().hex
    with jsonlines.open(path / f'{name_prefix}.stats.json', 'w', dumps=lambda x: json.dumps(x, cls=SetEncoder)) as f:
        f.write({
            'users': users,
            'change_types': change_types,
            'change_sizes': change_sizes,
            'diff_types': diff_types,
            'change_counts': change_counts,
        })


def fulfills_criteria(args, current_table):
    if len(current_table.rows) < args.min_row_count:
        return False

    current_header = current_table.header
    if len(current_header) > args.max_column_count:
        return False

    if any(len(c[0]) > args.max_column_name_length for c in current_header if c[0]):
        return False
    return True


if __name__ == '__main__':
    main()
