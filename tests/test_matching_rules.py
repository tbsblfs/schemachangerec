import math

import numpy as np
import pandas as pd
import snapshottest
from numpy import float32

from rust_dist import get_distances_chunks


class MyModel:
    def __init__(self, name):
        self.vectors = [np.array([1, 2, 3], dtype=float32), np.array([4, 5, 6], dtype=float32), np.array([7, 8, 9], dtype=float32), np.array([10, 11, 12], dtype=float32),
                        np.array([13, 14, 15], dtype=float32), np.array([16, 17, 18], dtype=float32), np.array([2, 2, 2], dtype=float32)]
        self.key_to_index = {'ENTITY/1': 0, 'ENTITY/2': 1, 'ENTITY/3': 2, 'ENTITY/4': 3, 'ENTITY/5': 4, 'ENTITY/6': 5, 'ENTITY/7': 6}


class MatchinegRulesTestCas(snapshottest.TestCase):


    def test_rust_distances(self):
        pages = [['1', '2'], ['1', '3', '4'], ['5', '6'], ['1', '2', '3', '4', '5', '6']]
        rules = pd.DataFrame({'pages': pages, 'rule_id': [1, 2, 3, 4]})

        matching_rules_df = pd.DataFrame({'pageTitle': ['1', '1', '2', '2', '3', '3', '4', '7'], 'rule_id': [1, 2, 1, 2, 1, 2, 4, 4]})
        matching_rules_df = matching_rules_df.join(rules.set_index('rule_id'), on='rule_id')

        model = MyModel('test')
        vector_dict = {page[7:].replace('_', ' '):
                           np.array(model.vectors[idx]) for page, idx in
                       model.key_to_index.items()}

        result = get_distances_chunks(matching_rules_df['pageTitle'], matching_rules_df['pages'], vector_dict, 2)


        np.testing.assert_almost_equal(result[7, 0], math.sqrt(2))
        distance = math.sqrt(2) + math.sqrt(29) + math.sqrt(110) + math.sqrt(245) + math.sqrt(434)
        np.testing.assert_almost_equal(result[7, 1], distance / 5, decimal=4)


        np.testing.assert_almost_equal(result[7, 2], 0.0011514108596749617)
        np.testing.assert_almost_equal(result[7, 3], 0.004766297340393066)

    def test_add_distances(self):
        pages = [['1', '2'], ['1', '3', '4'], ['5', '6'], ['1', '2', '3', '4', '5', '6']]
        rules = pd.DataFrame({'pages': pages, 'rule_id': [1, 2, 3, 4]})

        matching_rules_df = pd.DataFrame({'pageTitle': ['1', '1', '2', '2', '3', '3', '4', '7'], 'rule_id': [1, 2, 1, 2, 1, 2, 4, 4]})
        df = matching_rules_df.join(rules.set_index('rule_id'), on='rule_id')
        print(df)
        df_with_pages = df.loc[~df['pages'].isnull()]
        df_with_pages['pages'] = df_with_pages.apply(
            lambda x: [page for page in x['pages'] if page != x['pageTitle']], axis=1)
        model = MyModel('test')
        vector_dict = {page[7:].replace('_', ' '):
                           np.array(model.vectors[idx]) for page, idx in
                       model.key_to_index.items()}

        result = get_distances_chunks(df['pageTitle'], df['pages'], vector_dict, 2)
        self.assertMatchSnapshot(result, 'distances')