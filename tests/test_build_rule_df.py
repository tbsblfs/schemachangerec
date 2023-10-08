import pandas as pd
from snapshottest import TestCase

from schemarec.mining.build_rule_df import add_user_info


class RuleDFTestCase(TestCase):
    def test_add_user_info(self):
        rules = {'editors': [['1', '2'], ['1'], ['4', '6']], 'rule': [1, 2, 3]}

        users = {'user': ['1', '2', '3', '4', '5', '6'], 'count': [1, 2, 3, 4, 5, 6]}

        rules_df = pd.DataFrame(rules)
        users_df = pd.DataFrame(users)
        rules_df = add_user_info(users_df, rules_df)
        self.assertMatchSnapshot(rules_df, 'rules_with_user_info')