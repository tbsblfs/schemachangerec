import numpy as np
import pandas as pd
import snapshottest

from schemarec.ranking.learn_ranking import limit_examples, get_partial_and_full_result, mrr


class RankingTestCase(snapshottest.TestCase):
    def test_limit_examples(self):
        df = pd.DataFrame(columns=['key', 'score', 'confidence', 'rule_id'], data=[
            ['a', 0, 0.6, 1],
            ['a', 0, 0.3, 2],
            ['b', 0, 0.3, 2],
            ['b', 0, 0.2, 3],
            ['a', 1, 0.2, 3],
            ['b', 0, 0.4, 1],
            ])
        limited = limit_examples(df, 2)
        self.assertEqual(set(limited.index.tolist()), set([0, 2, 4, 5]))

        limited = limit_examples(df, 1)
        self.assertEqual(set(limited.index.tolist()), set([4, 5]))

    def test_limit_examples2(self):
        df = pd.DataFrame(columns=['key', 'score', 'confidence', 'rule_id'], data=[
            ['a', 0, 0.6, 3],
            ['a', 0, 0.3, 2],
            ['b', 0, 0.3, 2],
            ['b', 0, 0.2, 3],
            ['a', 1, 0.2, 3],
            ['b', 0, 0.4, 1],
            ])
        limited = limit_examples(df, 2)
        self.assertEqual(set(limited.index.tolist()), set([1, 2, 4, 5]))

        limited = limit_examples(df, 1)
        self.assertEqual(set(limited.index.tolist()), set([4, 5]))

    def test_get_partial_and_full_result(self):
        df = pd.DataFrame(columns=['key', 'label', 'rank', 'confrank'], data=[
            ['a', 0, 0, 0],
            ['a', 1, 5, 10],
            ['a', 1, 4, 11],
            ['b', 2, 1, 5],
                          ])
        df['key'] = df['key'].astype('category')
        df['key'] = df['key'].cat.set_categories(['a', 'b', 'c'])
        df = get_partial_and_full_result(df)
        self.assertMatchSnapshot(df, 'partial_and_full_result')

    def test_mrr(self):
        labels = [1, 0, 0, 1, 0, 0, 0, 1, 2, 0] * 10000
        predictions = [0.4, 0.4, 0.2, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1] * 10000
        groups = [3, 4, 3] * 10000
        res = mrr(np.array(labels, dtype='float32'), np.array(predictions, dtype='float64'), None, pd.Series(groups, dtype='int64'))
        self.assertAlmostEqual(res[1], 0.5833333333333334)

