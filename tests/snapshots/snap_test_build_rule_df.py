# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import GenericRepr, Snapshot


snapshots = Snapshot()

snapshots['RuleDFTestCase::test_add_user_info rules_with_user_info'] = GenericRepr('  editors  rule  user_count_>=1  user_count_>=2  user_count_>=4  user_count_>=5\n0  [1, 2]     1               1               1               0               0\n1     [1]     2               1               0               0               0\n2  [4, 6]     3               0               0               1               1')
