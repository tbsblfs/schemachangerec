import argparse
import logging
import warnings

import dateutil.parser
import jsonlines
import numpy as np
import pandas as pd
from jsonlines import jsonlines
from networkx import DiGraph
from tqdm import tqdm

from schemarec.util import tuplify

warnings.simplefilter(action='ignore', category=FutureWarning)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    parser.add_argument('--nodes', help='graph node file', type=str, required=True)
    parser.add_argument('--edges', help='graph edge file', type=str, required=True)
    parser.add_argument('--users', type=str, help='users', default=None)
    parser.add_argument('--output', type=str, help='output file', required=True)

    args = parser.parse_args()

    logging.info("Loading graph...")
    G = load_graph(args.nodes, args.edges)

    logging.info("Extracting node data...")
    df_nodes = pd.DataFrame(G.nodes(data=True), columns=['node', 'data'])
    df_nodes = extract_data(df_nodes, ['weight', 'survived', 'deleted', 'discarded'])

    # turn G.edges into a dataframe
    logging.info("Extracting edge data...")
    df_edges = pd.DataFrame(G.edges(data=True), columns=['source', 'target', 'data'])
    df_edges = extract_data(df_edges, ['weight', 'survived', 'deleted', 'discarded', 'editors', 'pages', 'revisions'],
                            edge=True)
    df_edges['pages'] = df_edges['pages'].apply(lambda x: [(p, v) for p, v in x.items()] if x else None)
    df_edges['revisions'] = df_edges['revisions'].apply(
        lambda x: np.array([dateutil.parser.parse(v).timestamp() for r, v in x]))

    logging.info("Calculating edge weights...")
    df_edges['source_edge_weight'] = df_edges.groupby('source')['weight'].transform('sum')
    df_edges['target_edge_weight'] = df_edges.groupby('target')['weight'].transform('sum')

    logging.info("Calculating edge confidence...")
    df_edges['confidence'] = df_edges['weight'] / df_edges['source_edge_weight']
    # drop edges with confidence < 0.001
    df_edges = df_edges[df_edges['confidence'] >= 0.001]

    # add node information to edges without including the 'node' column
    logging.info("Adding node information to edges...")
    df_edges = df_edges.merge(df_nodes, left_on='source', right_on='node', suffixes=('', '_source'), how='left').drop(
        'node', axis=1)
    df_edges = df_edges.merge(df_nodes, left_on='target', right_on='node', suffixes=('', '_target'), how='left').drop(
        'node', axis=1)

    if 'editors' in df_edges.columns:
        logging.info("Adding editor information to edges...")
        df_edges['editor_count'] = df_edges['editors'].apply(lambda x: len(x))

        if args.users:
            logging.info("Loading user data...")
            df_users = load_user_data(args.users)

            logging.info("Adding user information to edges...")
            df_edges = add_user_info(df_users, df_edges)
        df_edges.drop(columns=['editors'], inplace=True)

    logging.info("Saving edges...")
    df_edges.to_pickle(args.output)

    pd.set_option('display.max_columns', None)

    logging.info(df_edges)

    # print column types
    logging.info(df_edges.dtypes)

    # print nan counts
    logging.info(df_edges.isna().sum())


def load_user_data(user_file):
    users = []
    counts = []
    with jsonlines.open(user_file) as reader:
        for stat in reader:
            for user, count in stat['users'].items():
                users.append(user)
                counts.append(count)
    return pd.DataFrame({'user': users, 'count': counts})


def extract_data(df, fields, edge=False):
    # turn data into columns
    for field in fields:
        df[field] = df['data'].apply(lambda k: k[field] if field in k else None)

    if edge:
        df['single-revision'] = df['data'].apply(
            lambda k: k['revisions'][0][0] if 'revisions' in k and len(k['revisions']) == 1 else None)
        df['single-key'] = df['data'].apply(lambda k: k['keys'][0] if 'keys' in k and len(k['keys']) == 1 else None)

    # drop data column
    df.drop(columns=['data'], inplace=True)

    # fill weight, survived, deleted and discarded with 0 if they are NaN
    df['weight'] = df['weight'].fillna(0)
    df['survived'] = df['survived'].fillna(0)
    df['deleted'] = df['deleted'].fillna(0)
    df['discarded'] = df['discarded'].fillna(0)
    return df


def add_user_info(df_users, df_edges):
    df_users = df_users.sort_values(by='count', ascending=True)
    df_users['sum'] = df_users['count'].cumsum()
    total = df_users['sum'].iloc[-1]
    user_change_limits = [
        df_users[df_users['sum'] >= total * 0.67].iloc[0]['count'],
        df_users[df_users['sum'] >= total * 0.33].iloc[0]['count'],
        2,
        1,
    ]
    # get the first index for which the count is larger than the user_change_limit
    df_users['type'] = df_users['count'].apply(
        lambda x: next((">=" + str(v) for i, v in enumerate(user_change_limits) if v <= x), "==0"))
    # rule users
    rule_users = df_edges[['editors']].explode('editors').reset_index()
    rule_users = rule_users.merge(df_users[['user', 'type']], left_on='editors', right_on='user',
                                  suffixes=('', '_user'), how='left').drop('user', axis=1).set_index('index')

    # for each index, count the number of editors of each type and create a column for each type with the count
    rule_users = pd.get_dummies(rule_users['type'], prefix='user_count').groupby('index').sum()

    # add the columns to the edges dataframe
    df_edges = pd.concat([df_edges, rule_users], axis=1)

    return df_edges


def load_graph(nodes, edges):
    G = DiGraph()
    with jsonlines.open(nodes) as reader:
        G.add_nodes_from((tuplify(n), d) for n, d in tqdm(reader))

    with jsonlines.open(edges) as reader:
        G.add_edges_from((tuplify(u), tuplify(v), d) for u, v, d in tqdm(reader))

    return G


if __name__ == '__main__':
    main()
