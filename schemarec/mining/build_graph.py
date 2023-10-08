import argparse
import fileinput
import json
import uuid
from collections import defaultdict, Counter
from pathlib import Path

import networkx as nx
import tqdm as tqdm
from jsonlines import jsonlines

from schemarec.mining import execute_strategy

from schemarec.schemamatching import create_config, create_correspondences
from schemarec.util import SetEncoder, fulfills_criteria, TableFilter
from schemarec.table import Table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')
    parser.add_argument('--strategy', choices=['include', 'exclude', 'project', 'split', 'whole'],
                        help='strategy', default='include')
    parser.add_argument('--min-row-count', type=int, help='minimum row count', default=1)
    parser.add_argument('--max-column-count', type=int, help='maximum column count', default=100)
    parser.add_argument('--max-column-name-length', type=int, help='maximum column name length', default=150)
    parser.add_argument('--output', help='output folder', type=str, required=True)

    args = parser.parse_args()

    G = nx.DiGraph()

    config = create_config()

    table_filter = TableFilter(args.min_row_count, args.max_column_count, args.max_column_name_length)

    with jsonlines.Reader(fileinput.input(files=args.files if len(args.files) > 0 else ('-',))) as reader:
        for v in tqdm.tqdm(reader):
            if len(v) == 0:
                continue

            handle_table(G, table_filter, args.strategy, config, v)
    write_graph(G, args.output, args.files)


def handle_table(G, table_filter, strategy, matching_config, versions):
    table_header_subsequences = set()
    last_table_version_header_subsequences = set()
    table_header_changes = set()
    table_header_changes_editors = defaultdict(set)
    last_version_type = None
    page_name = versions[0]['pageTitle']
    key = versions[0]['key']
    for version in versions:
        current_version = version == versions[-1]

        if 'header' not in version:
            if current_version:
                last_version_type = "deleted"
            continue

        current_table = Table(version)
        if not table_filter.matches(current_table):
            if current_version:
                last_version_type = "discarded"
            continue

        for subseq in create_subsequences(current_table.header):
            if current_version:
                last_table_version_header_subsequences.add(subseq)
            table_header_subsequences.add(subseq)

        if 'previous-stable' not in version or 'header' not in version['previous-stable']:
            continue

        prev_table = Table(version['previous-stable'])
        if not table_filter.matches(prev_table):
            continue

        correspondences = create_correspondences(version, matching_config)
        for source, target in execute_strategy(correspondences, current_table.header, prev_table.header,
                                               strategy):
            table_header_subsequences.add(source)
            table_header_subsequences.add(target)

            G.add_edge(source, target)
            table_header_changes.add((source, target))
            if 'revisions' not in G[source][target]:
                G[source][target]['revisions'] = set()
            G[source][target]['revisions'].add((version['revisionId'], version['validFrom']))
            if 'user' in version and 'id' in version['user']:
                table_header_changes_editors[(source, target)].add(version['user']['id'])
    for subseq in table_header_subsequences:
        G.add_node(subseq)
        G.nodes[subseq]['weight'] = G.nodes[subseq].get('weight', 0) + 1
        if subseq in last_table_version_header_subsequences:
            G.nodes[subseq]['survived'] = G.nodes[subseq].get('survived', 0) + 1
        if last_version_type is not None:
            G.nodes[subseq][last_version_type] = G.nodes[subseq].get(last_version_type, 0) + 1
    for source, target in table_header_changes:
        G[source][target]['weight'] = G[source][target].get('weight', 0) + 1
        if target in last_table_version_header_subsequences:
            G[source][target]['survived'] = G[source][target].get('survived', 0) + 1
            if 'pages' not in G[source][target]:
                G[source][target]['pages'] = Counter()
            G[source][target]['pages'][page_name] += 1
        if 'editors' not in G[source][target]:
            G[source][target]['editors'] = Counter()
        G[source][target]['editors'].update(table_header_changes_editors[(source, target)])

        if 'keys' not in G[source][target]:
            G[source][target]['keys'] = set()
        G[source][target]['keys'].add(key)

        if last_version_type is not None:
            G[source][target][last_version_type] = G[source][target].get(last_version_type, 0) + 1

def write_graph(G, output, files):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    name_prefix = files[0].split('/')[-1].split('.json')[0] if len(files) == 1 else uuid.uuid4().hex
    with jsonlines.open(path / f'{name_prefix}.nodes.json', 'w', dumps=lambda x: json.dumps(x, cls=SetEncoder)) as f:
        for n in G.nodes(data=True):
            f.write(n)
    with jsonlines.open(path / f'{name_prefix}.edges.json', 'w', dumps=lambda x: json.dumps(x, cls=SetEncoder)) as f:
        for n in G.edges(data=True):
            f.write(n)


def create_subsequences(sequence):
    for length in range(1, len(sequence) + 1):
        for start in range(len(sequence) - length + 1):
            yield sequence[start:start + length]


if __name__ == '__main__':
    main()
