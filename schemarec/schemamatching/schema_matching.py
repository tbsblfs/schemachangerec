import Levenshtein
import networkx as nx
from networkx.algorithms.bipartite import minimum_weight_full_matching
from thefuzz import fuzz

from schemarec.table import Table


def create_correspondences(obj, config):
    previous_table = Table(obj['previous-stable'])
    current_table = Table(obj)
    graph = classify_renamings(previous_table, current_table)
    matchings = get_min_weight_matching(config, graph)

    for common_element in set(previous_table.header).intersection(set(current_table.header)):
        matchings.add(((common_element,), (common_element,)))
    return matchings


def create_renaming_graph(obj):
    return classify_renamings(Table(obj['previous-stable']), Table(obj), True)


def create_config(conf_content_score=0.4, conf_name_score=0.4, conf_min_length=2, conf_min_values=1, name_weight=0.5,
                  name_score_type="name_score", content_score_type="content_token_sort_score"):
    return conf_content_score, conf_name_score, conf_min_length, conf_min_values, name_weight, name_score_type, content_score_type


def get_min_weight_matching(config, graph):
    conf_content_score, conf_name_score, conf_min_length, conf_min_values, name_weight, name_score_type, content_score_type = config
    # iterate edges and calculate weight
    for (u, v, d) in graph.edges(data=True):
        if d[content_score_type] >= conf_content_score and d['min_token_count'] >= conf_min_values or \
                d[name_score_type] >= conf_name_score and d['min_name_length'] >= conf_min_length:
            d['weight'] = 1 - (
                    name_weight * d[name_score_type] + (1.0 - name_weight) * d[content_score_type])
        else:
            d['weight'] = 1000
    matching_edges = set()
    if len(graph.edges) > 0:
        min_weight_matchings = minimum_weight_full_matching(graph)
        # only keep edges with a weight lower than 1 in matchings
        for (snode, tnode) in min_weight_matchings.items():
            if graph[snode][tnode]['weight'] < 1:
                matching_edges.add((snode[1], tnode[1]) if snode[0] == "add" else (tnode[1], snode[1]))
    return matching_edges


def classify_renamings(previous_version, current_version, all_scores=False):
    current_header = current_version.header
    previous_header = previous_version.header

    # common elements
    common_elements = set(current_header).intersection(set(previous_header))

    # remove common elements from previous and current header positions
    added_positions = {v: k for k, v in enumerate(current_header) if v not in common_elements}
    removed_positions = {v: k for k, v in enumerate(previous_header) if v not in common_elements}

    matchings = nx.Graph()
    for added, added_idx in added_positions.items():
        added_column_string = current_version.column_string(added_idx)
        added_tokens = current_version.column_tokens(added_idx)

        for removed, removed_idx in removed_positions.items():
            removed_column_string = previous_version.column_string(removed_idx)
            removed_tokens = previous_version.column_tokens(removed_idx)

            data = dict()
            data['name_score'] = Levenshtein.ratio(removed[0], added[0])
            if all_scores:
                data['name_partial_score'] = fuzz.partial_ratio(removed[0], added[0]) / 100
                data['name_token_set_score'] = fuzz.token_set_ratio(removed[0], added[0]) / 100
                data['name_token_sort_score'] = fuzz.token_sort_ratio(removed[0], added[0]) / 100
                data['name_partial_token_set_score'] = fuzz.partial_token_set_ratio(removed[0], added[0]) / 100

            data['content_token_sort_score'] = fuzz.token_sort_ratio(added_column_string, removed_column_string) / 100
            if all_scores:
                data['content_score'] = jaccard_similarity(added_tokens, removed_tokens)
                data['content_partial_score'] = fuzz.partial_ratio(added_column_string, removed_column_string) / 100
                data['content_token_set_score'] = fuzz.token_set_ratio(added_column_string, removed_column_string) / 100
                data['content_partial_token_set_score'] = fuzz.partial_token_set_ratio(added_column_string,
                                                                                       removed_column_string) / 100

            data['min_token_count'] = min(len(added_tokens), len(removed_tokens))
            data['min_name_length'] = min(len(removed[0]), len(added[0])) if removed[0] and added[0] else 0
            matchings.add_edge(("add", (added,)), ("rem", (removed,)), **data)

    return matchings


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / max(len(s1.union(s2)), 1))
