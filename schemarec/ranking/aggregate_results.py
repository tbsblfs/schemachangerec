import argparse
import fileinput
import json

import dateutil
import jsonlines
import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from schemarec.util import fulfills_criteria
from schemarec.table import Table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', '-r', help='result file', required=True)
    parser.add_argument('--type', default='test')
    parser.add_argument('--min-row-count', type=int, help='minimum row count', default=1)
    parser.add_argument('--max-column-count', type=int, help='maximum column count', default=100)
    parser.add_argument('--max-column-name-length', type=int, help='maximum column name length', default=150)
    parser.add_argument('--correlation', help='correlation', default=False, action='store_true')
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    args = parser.parse_args()

    change_dates = dict()
    diff_sizes = dict()
    row_sizes = dict()
    schema_sizes = dict()
    with jsonlines.Reader(fileinput.input(files=args.files if len(args.files) > 0 else ('-',))) as reader:
        for table in tqdm(reader):
            if len(table) == 0:
                continue

            valid_changes = [version for version in table if 'header' in version and
                             'previous-stable' in version and
                             'header' in version['previous-stable'] and
                             fulfills_criteria(args, Table(version))
                             and fulfills_criteria(args, Table(version['previous-stable']))]

            if len(valid_changes) == 0:
                continue

            first_version = next(iter(valid_changes))
            key = str(first_version['key'])
            change_dates[key] = dateutil.parser.parse(first_version['validFrom'])
            header_set_previous = set(first_version['previous-stable']['header'])
            header_set_current = set(first_version['header'])
            diff_sizes[key] = len(header_set_previous.symmetric_difference(header_set_current))
            if 'rows' in first_version:
                row_sizes[key] = first_version['rows']
            if 'columns' in first_version:
                schema_sizes[key] = first_version['columns']

    df = DataFrame.from_dict(change_dates, orient='index', columns=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    df['rows'] = pd.Series(row_sizes)
    df['columns'] = pd.Series(schema_sizes).map(log_bins)
    df['diff_sizes'] = pd.Series(diff_sizes).map(log_bins)
    df['rows'] = df['rows'].map(log_bins)
    df = df.reset_index(names=['key'])

    # read json lines to pandas
    df_results = pd.read_json(args.result, lines=True)
    df_results = df_results[df_results['set'] == args.type]

    # group df by set and dropped-column (include null)
    if 'dropped-columns' not in df_results:
        df_results['dropped-columns'] = '[None]'

    df_results['dropped-columns'] = df_results['dropped-columns'].fillna('[None]')
    df_results['dropped-columns'] = df_results['dropped-columns'].astype(str)
    grouped = df_results.groupby('dropped-columns')

    # merge each group with df
    grouped = {x: df.merge(grouped.get_group(x), on="key", how="left") for x in grouped.groups}

    # for each group calculate mrr of rank_partial
    for scenario in ['partial', 'full']:
        for rank in ['rank', 'confrank']:

            field = f"{rank}_{scenario}"

            for selection_field in [None, 'month', 'rows', 'columns', 'diff_sizes']:
                for selection_value in list(df[selection_field].unique()) if selection_field is not None else [None]:
                    subset_getter = lambda group: group
                    if selection_field is not None:
                        subset_getter = lambda group: group[get_subset(group, selection_field, selection_value)]

                    for obj in get_output(field, grouped, subset_getter):
                        obj['set'] = args.type
                        obj['rank'] = rank
                        obj['scenario'] = scenario
                        obj['selection'] = selection_field
                        obj['selection_value'] = str(selection_value)
                        print(json.dumps(obj))

                    if args.correlation and selection_field is not None:
                        for selection_field2 in ['month', 'rows', 'columns', 'diff_sizes']:
                            if selection_field2 <= selection_field:
                                continue
                            for selection_value2 in list(
                                    df[selection_field].unique()) if selection_field is not None else [None]:
                                subset_getter = lambda group: group[
                                    get_subset(group, selection_field, selection_value) & get_subset(group,
                                                                                                     selection_field2,
                                                                                                     selection_value2)]
                                for obj in get_output(field, grouped, subset_getter):
                                    obj['set'] = args.type
                                    obj['rank'] = rank
                                    obj['scenario'] = scenario
                                    obj['selection'] = f"{selection_field}+{selection_field2}"
                                    obj['selection_value'] = str(selection_value)
                                    obj['selection_value2'] = str(selection_value2)
                                    print(json.dumps(obj))


def get_output(field, grouped, subset_getter):
    none_group = grouped['[None]']
    none_group = subset_getter(none_group)
    none_mrr = 1 / none_group[field]
    none_mrr_mean = none_mrr.mean()
    if math.isnan(none_mrr_mean):
        none_mrr_mean = 0
    for name, group in grouped.items():
        group = subset_getter(group)
        if len(group) == 0:
            continue

        count, mrr, recall, recall_at_k = evaluate(field, group)

        yield {
            'dropped-column': name,
            'mrr': mrr,
            'none_mrr': none_mrr_mean,
            'recall': recall,
            'recall_at_k': [{'k': k, 'count': v} for k, v in recall_at_k.to_dict().items()],
            'count': count,
        }


def log_bins(x):
    if x is None or np.isnan(x):
        return None

    if x < 3:
        return str(x)

    return str(math.pow(2, int(math.log(x - 1, 2))) + 1) + " - " + str(math.pow(2, int(math.log(x - 1, 2)) + 1))


def evaluate(field, group):
    group = group.sort_values(field)
    group['rank_mrr'] = 1 / group[field]
    # calculate recall@k
    # counter number of keys with rank <= k for each k
    recall_at_k = group[field].value_counts().sort_index().cumsum()
    # count NaNs in field
    recall = 1 - group[field].isna().sum() / len(group)
    mrr = group['rank_mrr'].mean()
    if math.isnan(mrr):
        mrr = 0
    count = len(group)
    return count, mrr, recall, recall_at_k


def get_subset(group, selection_field, selection_value):
    return group[selection_field] == selection_value


if __name__ == '__main__':
    main()
