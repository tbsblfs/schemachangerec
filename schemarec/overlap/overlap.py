import argparse
import fileinput
import warnings

import jsonlines
import pandas as pd
import tqdm

from schemarec.mining import execute_strategy
from schemarec.schemamatching import create_correspondences, create_config
from schemarec.util import fulfills_criteria, TableFilter
from schemarec.table import Table
from .matching_rules import StatsHandler

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    parser.add_argument('--rules', help='graph rules file', type=str, required=True)
    parser.add_argument('--reordering_strategy', choices=['include', 'exclude', 'project', 'split'],
                        help='reordering strategy', default='include')
    parser.add_argument('--min-row-count', type=int, help='minimum row count', default=1)
    parser.add_argument('--max-column-count', type=int, help='maximum column count', default=100)
    parser.add_argument('--max-column-name-length', type=int, help='maximum column name length', default=150)
    parser.add_argument('--embedding', type=str, help='embedding', default=None)
    parser.add_argument('--ignore-own-page', help='ignore own page', default=False, action='store_true')
    parser.add_argument('--output', type=str, help='output', default=None)

    args = parser.parse_args()

    rules = pd.read_pickle(args.rules)

    model = None
    if args.embedding:
        import gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(args.embedding, no_header=True, binary=False)

    handler = StatsHandler(rules, model, args.output, args.ignore_own_page)
    table_filter = TableFilter(args.min_row_count, args.max_column_count, args.max_column_name_length)
    matching_config = create_config()
    with jsonlines.Reader(fileinput.input(files=args.files if len(args.files) > 0 else ('-',))) as reader:
        for table in tqdm.tqdm(reader):
             handle_schema_changes(table, handler, matching_config, args.reordering_strategy, table_filter)
    handler.handle_complete()


def handle_schema_changes(table, handler, matching_config, reordering_strategy, table_filter):
    handler.handle_new_table(table)

    for version in table:
        if not 'header' in version:
            continue

        if not 'previous-stable' in version or not 'header' in version['previous-stable']:
            continue

        current_table = Table(version)
        prev_table = Table(version['previous-stable'])

        if not table_filter.matches(current_table) or not table_filter.matches(prev_table):
            handler.handle_invalid_change(prev_table, current_table)
            continue

        correspondences = create_correspondences(version, matching_config)
        gen = execute_strategy(correspondences, current_table.header, prev_table.header,
                               reordering_strategy)
        if not handler.handle_valid_change(prev_table, current_table, gen, version, correspondences):
            break

    handler.handle_table_done()



if __name__ == '__main__':
    main()
