# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import GenericRepr, Snapshot


snapshots = Snapshot()

snapshots['RankingTestCase::test_get_partial_and_full_result partial_and_full_result'] = GenericRepr('  key  rank_partial  confrank_partial  rank_full  confrank_full\n0   a           4.0              10.0        NaN            NaN\n1   b           1.0               5.0        1.0            5.0\n2   c           NaN               NaN        NaN            NaN')
