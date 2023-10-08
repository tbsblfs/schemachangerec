import logging

import pandas as pd


def main():
    # read json from output/matching-eval/eval.json
    df = pd.read_json("output/matching-eval/eval.json", lines=True)
    df['changesize'] = df['group'].apply(lambda x: f"{2 ** (x - 1)} -- {2 ** x - 1}" if x > 1 else "1")
    logging.info(df)
    # sort by group
    df = df.sort_values(by=['group'])
    df = df[['changesize', 'precision', 'recall', 'f1']]
    # use changesize as index
    df = df.set_index('changesize')
    s = df.style

    s = s.format(precision=3)
    s = s.format_index(escape="latex", axis=0, precision=0)

    print(s.to_latex(siunitx=True))


if __name__ == '__main__':
    main()
