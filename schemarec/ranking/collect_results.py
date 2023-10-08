import argparse
import json
import logging
import os
import sys

import jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='input dir')
    args = parser.parse_args()

    for testscenario in os.listdir(args.input):
        scenario_path = os.path.join(args.input, testscenario)

        # if testscenario is not a dir skip
        if not os.path.isdir(scenario_path):
            continue

        for testset in os.listdir(scenario_path):
            combo_path = os.path.join(scenario_path, testset)
            if not os.path.isdir(combo_path):
                continue

            if testset == 'train' or testset == 'test':
                continue

            # load aggregated results
            for method in ['mining', 'baseline']:
                for model in ['lgbm', 'flaml']:
                    result_file = os.path.join(combo_path, f'ranking-{method}-{model}-agg.json')
                    if not os.path.exists(result_file):
                        logging.warning(f'WARNING: {result_file} does not exist')
                        continue

                    with jsonlines.open(result_file) as f:
                        for obj in f:
                            obj['testscenario'] = testscenario
                            obj['testset'] = testset
                            obj['method'] = method
                            obj['model'] = model
                            print(json.dumps(obj))


if __name__ == '__main__':
    main()
