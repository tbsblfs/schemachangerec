import argparse
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from rust_dist import get_mrr

from schemarec.util import search_files

TARGET_FIELD = 'score'
TRAIN_TEST_SPLIT_KEY = 'pageTitle'
GROUP_ATTRIBUTE = 'key'
EVALUATION_GROUP = 'key'

feature_groups = [
    ['precondition', 'precondition_length', 'postcondition', 'postcondition_length'],
    ['pageTitle', 'revisionId', 'key'],
    ['rule_id']
]


def turn_to_string(header):
    if header[1] == 0: return str(header[0])
    return f"{header[0]} ({header[1]})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', help='input file', required=True)
    parser.add_argument('--test', help='test input file')
    parser.add_argument('--training_ratio', '-r', help='ratio of training data', default=0.8)
    parser.add_argument('--objective', '-t', help='objective', default='lambdarank',
                        choices=['lambdarank', 'rank_xendcg'])
    parser.add_argument('--output', '-o', help='output file', required=True)
    parser.add_argument('--output-rules', help='output rules')
    parser.add_argument('--test-changes')
    parser.add_argument('--learning-rate', help='learning rate', default=0.1, type=float)
    parser.add_argument('--n-estimators', help='number of estimators', default=100, type=int)
    parser.add_argument('--num-leaves', help='number of leaves', default=31, type=int)
    parser.add_argument('--min-data-in-leaf', help='min data in leaf', default=20, type=int)
    parser.add_argument('--boosting-type', help='boosting type', default='gbdt', choices=['gbdt', 'dart', 'rf'])
    parser.add_argument('--flaml', help='use flaml', action='store_true')
    parser.add_argument('--time-budget', help='time budget for flaml', default=10, type=int)
    parser.add_argument('--skip-transform', help='skip transform', action='store_true')
    parser.add_argument('--true-changes', type=str, help='true changes')
    parser.add_argument('--column-drop', action='store_true', help='drop columns')
    parser.add_argument('--column-group-drop', action='store_true', help='drop column groups')
    parser.add_argument('--full-only', action='store_true', help='only use full matches in training')
    args = parser.parse_args()

    logging.info("Reading data")

    true_changes = {}
    if args.test_changes:
        search_files(args.test_changes, lambda x: true_changes.update({x[0]['key']: x}) if len(x) > 0 else None)

    if args.test:
        train = pd.read_pickle(args.input)
        # print train memory usage for each column
        logging.info(train.memory_usage(deep=True))

        logging.info("Original train size: ", len(train))
        # train = pd.DataFrame(train[~((train["weight"] == 1) & (train["score"] > 0))])
        train = limit_examples(train)
        if args.full_only:
            train['score'] = train['score'].apply(lambda x: 1 if x == 2 else 0)
        logging.info("Limited train size: ", len(train))

        test = pd.read_pickle(args.test)
        logging.info(test.memory_usage(deep=True))
        logging.info("Original test size: ", len(test))
        test = limit_examples(test)
        logging.info("Limited test size: ", len(test))

        common_cols = set(train.columns).intersection(set(test.columns))
        for col in common_cols:
            if train[col].dtype == 'category' and test[col].dtype == 'category' and not col == 'key':
                uc = union_categoricals([train[col], test[col]])
                train[col] = pd.Categorical(train[col], categories=uc.categories)
                test[col] = pd.Categorical(test[col], categories=uc.categories)

        logging.info("Train column types", train.dtypes)
        logging.info("Test column types", test.dtypes)
        train = train[common_cols]
        test = test[common_cols]
    else:
        df = pd.read_pickle(args.input)
        logging.info("Performing train/test split")
        train, test = perform_train_test_split(df, args.training_ratio)
        train = limit_examples(train)
        test = limit_examples(test)

    train["key_str"] = train["key"]
    test["key_str"] = test["key"]
    test["precondition_str"] = test["precondition"]
    test["postcondition_str"] = test["postcondition"]
    # turn categorical columns into category ids
    for col in train.columns:
        if train[col].dtype == 'category' and not col.endswith('_str'):
            train[col] = train[col].cat.codes
            test[col] = test[col].cat.codes

    logging.info("Train memory usage after column conversion:")
    logging.info(train.memory_usage(deep=True))
    logging.info("Test memory usage after column conversion:")
    logging.info(test.memory_usage(deep=True))

    logging.info("Transforming data")
    X_train, y_train, group_train = transform(train)
    X_test, y_test, group_test = transform(test)

    if args.flaml:
        # replace Inf by 1000 in df X_train
        X_train = X_train.replace([np.inf], 1000).replace([-np.inf], -1000)
        X_test = X_test.replace([np.inf], 1000).replace([-np.inf], -1000)

    # for each column in X_train
    total_dfs = []
    model_infos = []
    to_drop = [[None]]
    if args.column_drop:
        to_drop += [[x] for x in list(X_train.columns)]
    if args.column_group_drop:
        # get all columns that start with user_ or editor_
        user_cols = [x for x in list(X_train.columns) if x.startswith('user_') or x.startswith('editor_')]
        feature_groups.append(user_cols)
        to_drop += feature_groups
    for cols in to_drop:
        if not all([(x is None) or (x in list(X_train.columns)) for x in cols]):
            logging.info("Skipping", cols, "as it is not a subset of", list(X_train.columns))
            continue
        df, model_info = evaluate_without_col(X_train, X_test, y_train, y_test, group_train, group_test, args, cols,
                                              true_changes)
        total_dfs.append(df)
        model_info['dropped_columns'] = cols
        model_infos.append(model_info)
    total_df = pd.concat(total_dfs)
    model_df = pd.DataFrame.from_dict(model_infos, orient='columns')
    total_df.to_json(args.output, orient='records', lines=True)
    model_df.to_json(args.output + '.model', orient='records', lines=True)


def limit_examples(df, limit=10000):
    # for each rule_id and GROUP_ATTRIBUTE, keep the one with the highest TARGET_FIELD
    # this translates to: it is enough to see a match in any revision to consider the rule a match
    df = df.sort_values(by=[GROUP_ATTRIBUTE, TARGET_FIELD, 'confidence'], ascending=[True, False, False]) \
        .groupby([GROUP_ATTRIBUTE, 'rule_id'], sort=False).head(1) \
        .groupby(GROUP_ATTRIBUTE, sort=False).head(
        limit)  # only keep top 10000 (according to confidence) rules per page

    return df


def evaluate_without_col(X_train, X_test, y_train, y_test, group_train, group_test, args, cols, true_changes):
    to_remove = ['key', 'key_str', 'precondition_str', 'postcondition_str']
    if any(cols):
        to_remove += cols
    logging.info("Dropping columns", cols, "from training and test set")
    # get df without col
    X_train_without_col = X_train[X_train.columns.difference(to_remove)]
    X_test_without_col = X_test[X_test.columns.difference(to_remove)]
    predictions_train, predictions_test, model_info = get_results(X_train_without_col, X_test_without_col, y_train,
                                                                  y_test,
                                                                  group_train, group_test, args)
    logging.info("Run Evaluation on train set")
    df_train = get_partial_and_full_result(evaluate(predictions_train, X_train, y_train, group_train))
    df_train['set'] = 'train'
    logging.info("Run Evaluation on test set")
    df_test = evaluate(predictions_test, X_test, y_test, group_test)
    if args.output_rules:
        test_rules = df_test
    df_test = get_partial_and_full_result(df_test)
    df_test['set'] = 'test'
    logging.info("Writing results to", args.output)
    df = pd.concat([df_train, df_test])
    df['dropped-columns'] = [cols for _ in range(len(df))]
    # for each group print the rules to html
    if args.output_rules:
        test_rules['precondition'] = X_test['precondition_str']
        test_rules['postcondition'] = X_test['postcondition_str']
        if args.true_changes:
            true_changes = pd.read_pickle(args.true_changes)
            test_rules = test_rules.merge(true_changes, on=['key'], how='left')

        logging.info("Writing rules to", args.output_rules)
        p = Path(args.output_rules)
        p.mkdir(parents=True, exist_ok=True)

        # for each non-empty group save-html
        test_rules.groupby(EVALUATION_GROUP).apply(
            lambda group: save_html(p, group, true_changes) if not group.empty else None)
    return df, model_info


def get_results(X_train, X_test, y_train, y_test, group_train, group_test, args):
    logging.info("Training model")
    start_time = time.time()
    feature_importance = None
    if args.flaml:
        from flaml import AutoML
        model = AutoML(skip_transform=args.skip_transform)

        model.fit(X_train, y_train, X_val=X_test, y_val=y_test,
                  groups_val=group_test, groups=group_train,
                  task='rank', time_budget=args.time_budget)

        if model.feature_importances_ is not None:
            feature_importance = {
                'split': {
                    key: float(value) for key, value in zip(X_train.columns, model.feature_importances_)
                }
            }
        else:
            feature_importance = None
    else:
        model = lgb.LGBMRanker(
            objective=args.objective,
            learning_rate=args.learning_rate,
            n_estimators=args.n_estimators,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            boosting_type=args.boosting_type,
        )
        model.fit(X_train, y_train, group=group_train, eval_set=[(X_test, y_test)], eval_group=[group_test],
                  eval_at=[1, 5],
                  verbose=True, eval_metric=['ndcg', mrr])

        feature_importance = {
            'split': {
                key: float(value) for key, value in
                zip(X_train.columns, model.booster_.feature_importance(importance_type='split'))
            },
            'gain': {
                key: float(value) for key, value in
                zip(X_train.columns, model.booster_.feature_importance(importance_type='gain'))
            }
        }
    training_time = time.time() - start_time
    logging.info("Training time:", training_time)

    model_info = {
        'model': 'flaml' if args.flaml else 'lgbm',
        'training_time': training_time,
        'feature_importance': feature_importance,
        'best-config': model.best_config if args.flaml else '',
    }
    logging.info("Evaluation on train set")
    start_time = time.time()
    predictions_train = model.predict(X_train)
    logging.info("Evaluation time:", time.time() - start_time)
    logging.info("Evaluation on test set")
    start_time = time.time()
    predictions_test = model.predict(X_test)
    logging.info("Evaluation time:", time.time() - start_time)
    return predictions_train, predictions_test, model_info


def save_html(path, group, true_changes):
    output = ""
    output += "<html><head><style>table {border-collapse: collapse; width: 50%; float: left; overflow: scroll; display: inline-block} table, th, td {border: 1px solid black;}</style></head><body>"
    output += "<h1>Page: {}</h1>".format(group.name)
    if group.name in true_changes:
        for version in true_changes[group.name]:
            output += "<hr>"
            if 'previous-stable' in version:
                output += "<h2>Revision: {}</h2>".format(version['previous-stable']['revisionId'])
                if 'header' in version['previous-stable']:
                    output += "<h3>Header: {}</h3>".format(version['previous-stable']['header'])
                if 'content' in version['previous-stable']:
                    output += version['previous-stable']['content']
            output += "<h2>Revision: {}</h2>".format(version['revisionId'])
            if 'header' in version:
                output += "<h3>Header: {}</h3>".format(version['header'])
            if 'content' in version:
                output += version['content']
    if 'true_change' in group.columns:
        output += "<h2>True change:</h2>"
        output += f"<p>{','.join(map(turn_to_string, group.iloc[0]['true_change'][0]))} -> {','.join(map(turn_to_string, group.iloc[0]['true_change'][1]))}</p>"
        output += "<h2>Projected change:</h2>"
        for change in group.iloc[0]['projected_change']:
            output += f"<p>{','.join(map(turn_to_string, change[0]))} -> {','.join(map(turn_to_string, change[1]))}</p>"

    output += "<hr><h2>Predicted changes:</h2>"
    output += render_table(group, 'rank')
    output += render_table(group, 'confrank')
    output += "</body></html>"

    # does the group contain any rules that fully match?
    name = ""
    if group[group['label'] == 2].shape[0] > 0:
        # compare the ranks of the rules that fully match
        diff = group[group['label'] == 2]['rank'].min() - group[group['label'] == 2]['confrank'].min()
        name += "b" if diff < 0 else "w" if diff > 0 else "s"
    else:
        name += "n"
    if group[group['label'] > 0].shape[0] > 0:
        # compare the ranks of the rules that fully match
        diff = group[group['label'] > 0]['rank'].min() - group[group['label'] > 0]['confrank'].min()
        name += "b" if diff < 0 else "w" if diff > 0 else "s"
    else:
        name += "n"
    (path / name).mkdir(parents=True, exist_ok=True)

    with open(path / name / f'{safe_name(group.name)}.html', 'w') as f:
        f.write(output)


def render_table(group, criterion):
    df = group.sort_values(by=criterion)
    # get those with rank <= 10 or label > 0
    df = df[(df[criterion] <= 10) | (df['label'] > 0)]
    df.insert(0, criterion, df.pop(criterion))
    df.insert(1, 'label', df.pop('label'))
    df.insert(2, 'precondition', df.pop('precondition'))
    df.insert(3, 'postcondition', df.pop('postcondition'))
    style = df.style.apply(
        lambda x: [
            'background-color: green;' if x['label'] == 2 else 'background-color: yellow;' if x['label'] == 1 else ''
            for _ in
            x], axis=1)
    return style.to_html()


def safe_name(name):
    return "".join([c for c in name if c.isalpha() or c.isdigit() or c == ' ']).rstrip()


def evaluate(predictions, X, y, group):
    logging.info("mrr:", mrr(y.to_numpy(), predictions, None, group)[1])

    start = time.time()
    # rank rules within each group
    df = pd.DataFrame({'key': X['key_str'], 'label': y, 'prediction': predictions, 'confidence': X['confidence']})
    df = df.sort_values(by=[EVALUATION_GROUP, 'prediction', 'label'], ascending=[True, False, True])
    # minimum rank by prediction for each group with ties broken by label
    df['rank'] = df.groupby(EVALUATION_GROUP, sort=False).cumcount() + 1
    df['rank'] = df.groupby([EVALUATION_GROUP, 'prediction', 'label'], sort=False)['rank'].transform('min')

    df = df.sort_values(by=[EVALUATION_GROUP, 'confidence', 'label'], ascending=[True, False, True])
    df['confrank'] = df.groupby(EVALUATION_GROUP, sort=False).cumcount() + 1
    df['confrank'] = df.groupby([EVALUATION_GROUP, 'confidence', 'label'], sort=False)['confrank'].transform('min')
    logging.info("evaluate:", time.time() - start, "seconds")

    # only keep columns that are relevant for the evaluation
    return df[[EVALUATION_GROUP, 'label', 'rank', 'confrank']]


def get_partial_and_full_result(df):
    # get the minimum rank and confrank for each group where the label is not 0
    start = time.time()
    df_partial = df[df['label'] != 0].groupby(EVALUATION_GROUP)[['rank', 'confrank']].min().reset_index()
    df_full = df[df['label'] == 2].groupby(EVALUATION_GROUP)[['rank', 'confrank']].min().reset_index()
    df = df_partial.merge(df_full, on=EVALUATION_GROUP, suffixes=('_partial', '_full'), how='outer')
    logging.info("get_partial_and_full_result:", time.time() - start, "seconds")
    return df


def mrr(labels, predictions, weights, groups):
    start = time.time()
    labels = labels.astype(np.int32)
    predictions = predictions.astype(np.float64)
    groups = groups.astype(np.int64).to_numpy()
    mean_rank = get_mrr(labels, predictions, groups)
    logging.info("mrr:", mean_rank, "in", time.time() - start, "seconds")

    return 'mrr', mean_rank, True


def perform_train_test_split(df, ratio):
    g = df.groupby([TRAIN_TEST_SPLIT_KEY])
    a = np.arange(g.ngroups)
    np.random.shuffle(a)
    train_size = int(g.ngroups * ratio)
    train = df[g.ngroup().isin(a[:train_size])].reset_index(drop=True)
    test = df[g.ngroup().isin(a[train_size:])].reset_index(drop=True)
    train["key"] = train["key"].cat.remove_unused_categories()
    test["key"] = test["key"].cat.remove_unused_categories()
    return train, test


def transform(df):
    # sort dataframe by GROUP_ATTRIBUTE
    df = df.sort_values(by=[GROUP_ATTRIBUTE])
    # group by GROUP_ATTRIBUTE and get length of each group
    group = df.groupby(GROUP_ATTRIBUTE).size()
    X = df[df.columns.difference([TARGET_FIELD])]
    y = df[TARGET_FIELD]
    return X, y, group


if __name__ == '__main__':
    main()
