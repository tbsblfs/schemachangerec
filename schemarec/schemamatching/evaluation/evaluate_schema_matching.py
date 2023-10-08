import fileinput
import logging
from collections import defaultdict
from pathlib import Path

from analysis.schemamatching.schema_matching import get_min_weight_matching, create_renaming_graph
from analysis.util.util import tuplify
from jsonlines import jsonlines
from tqdm import tqdm

input = Path("//data/column-gold/")


class ResultStore:
    tp = 0
    fp = 0
    fn = 0
    fp_examples = []
    fn_examples = []


def main():
    configs = create_configs()

    data = load_data()
    gold = load_gold()
    finished = load_finished()
    # out_path = Path("output/" + time.strftime("%Y%m%d-%H%M%S") + "/")
    out_path = Path("output/matching-eval/")
    out_path.mkdir(parents=True, exist_ok=True)

    print_examples = False

    with open(out_path / "eval.json", 'w') as f:
        with jsonlines.Writer(f) as writer:
            for group, samples in data.items():
                if not any(s['revisionId'] in finished for s in samples):
                    continue

                logging.info("Handling group: ", group)

                results = defaultdict(ResultStore)

                for obj in tqdm(samples):
                    if not obj['revisionId'] in finished:
                        continue

                    # gold_current_revision = [g for g in gold if g['revisionId'] == obj['revisionId'] and len(g['added']) == 1 and len(g['removed']) == 1]
                    gold_matchings = get_gold_matching(gold, obj['revisionId'])

                    graph = create_renaming_graph(obj)
                    if graph.number_of_edges() > 0:
                        for config in configs:
                            matching_edges = get_min_weight_matching(config, graph)
                            res = results[config]
                            evaluate(gold_matchings, group, matching_edges, obj, print_examples, res)

                for config in configs:
                    write_results(config, group, results, writer)


def evaluate(gold_matchings, group, matching_edges, obj, print_examples, res):
    tp_local = len(set(matching_edges).intersection(gold_matchings))
    res.tp += tp_local
    res.fp += len(matching_edges) - tp_local
    res.fn += len(gold_matchings) - tp_local
    fp_matchings = set(matching_edges).difference(gold_matchings)
    if print_examples and len(fp_matchings) > 0:
        logging.info("Group: ", group, "Revision: ", obj['revisionId'])
        logging.info("FP: ", fp_matchings)
    if len(fp_matchings) > 0 and len(res.fp_examples) < 10:
        example = next(iter(fp_matchings))
        res.fp_examples.append(dict(
            revisionId=obj['revisionId'],
            added=example[0],
            removed=example[1],
        ))
    fn_matchings = set(gold_matchings).difference(matching_edges)
    if print_examples and len(fn_matchings) > 0:
        logging.info("Group: ", group, "Revision: ", obj['revisionId'])
        logging.info("FN: ", fn_matchings)
    if len(fn_matchings) > 0 and len(res.fn_examples) < 10:
        example = next(iter(fn_matchings))
        res.fn_examples.append(dict(
            revisionId=obj['revisionId'],
            added=example[0],
            removed=example[1],
        ))


def write_results(config, group, results, writer):
    tp = results[config].tp
    fp = results[config].fp
    fn = results[config].fn
    writer.write({
        "group": group,
        "config": config,
        "content_score": config[0],
        "name_score": config[1],
        "min_length": config[2],
        "min_values": config[3],
        "name_weight": config[4],
        "name_score_type": config[5],
        "content_score_type": config[6],
        "tp": results[config].tp,
        "fp": results[config].fp,
        "fp_examples": results[config].fp_examples,
        "fn": results[config].fn,
        "fn_examples": results[config].fn_examples,
        "precision": tp / (tp + fp) if tp + fp > 0 else 1,
        "recall": tp / (tp + fn) if tp + fn > 0 else 1,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 1
    })


def get_gold_matching(gold, revisionId):
    gold_current_revision = [g for g in gold if g['revisionId'] == revisionId]
    gold_matchings = set((g['added'], g['removed']) for g in gold_current_revision)
    return gold_matchings


def create_configs():
    content_scores = [0.2, 0.4, 0.6, 0.8]
    content_score_types = ["content_score", "content_partial_score", "content_token_set_score",
                           "content_token_sort_score", "content_partial_token_set_score"]
    name_scores = [0.2, 0.4, 0.6, 0.8]
    name_score_types = ["name_score", "name_partial_score", "name_token_set_score", "name_token_sort_score",
                        "name_partial_token_set_score"]
    name_weight = [0, 0.25, 0.5, 0.75, 1.0]
    min_length = [1, 2, 5, 10]
    min_values = [1, 2, 5, 10]
    content_scores = [0.2]
    name_scores = [0.2]
    content_score_types = ["content_token_sort_score"]
    name_score_types = ["name_score"]
    min_values = [1]
    min_length = [2]
    name_weight = [0.5]
    # cross product of scores
    configs = [(c, n, l, v, nw, nt, ct) for c in content_scores for n in name_scores for l in min_length for v in
               min_values for nw in name_weight for nt in name_score_types for ct in content_score_types]
    return configs


def load_data():
    data = {}
    with jsonlines.Reader(fileinput.input(files=input / "sample_strat.json")) as reader:
        for s in reader:
            data = s
    return data


def load_gold():
    with jsonlines.Reader(fileinput.input(files=input / "gold.json")) as reader:
        return tuplify([s for s in reader])


def load_finished():
    with jsonlines.Reader(fileinput.input(files=input / "finished.json")) as reader:
        return set(s['revisionId'] for s in reader if not s['skipped'])


if __name__ == '__main__':
    main()
