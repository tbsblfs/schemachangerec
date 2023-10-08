import logging

import dateutil.parser
import numpy as np
import pandas as pd
from rust_dist import get_minimum_timediff

from schemarec.util import catchtime_local
from .change_handler import ChangeHandler


def get_required_target(prev_table, current_table, start, length):
    if not prev_table.header[:start] == current_table.header[:start]:
        return None
    suffix_prev = prev_table.header[start + length:]
    suffix_current = current_table.header[-len(suffix_prev):] if len(suffix_prev) > 0 else ()
    if suffix_prev != suffix_current:
        return None
    required_target = current_table.header[start:-len(suffix_current)] if len(
        suffix_current) > 0 else current_table.header[start:]
    return required_target


class StatsHandler(ChangeHandler):
    def __init__(self, rules, embedding_model, output, ignore_own_page=False):
        self.stats = {}
        self.embedding_model = embedding_model
        self.rules = rules

        # create unique ids for each source
        with catchtime_local("Calculating source ids"):
            source_cat = self.rules['source'].astype('category').cat
        with catchtime_local("Setting source ids"):
            self.rules['source_id'] = source_cat.codes
        # add index as column 'rule_id'
        self.rules['rule_id'] = self.rules.index
        with catchtime_local("Setting index"):
            self.rules.set_index('source_id', inplace=True)
        with catchtime_local("Sorting"):
            self.rules.sort_index(inplace=True)

        # create dict from source to source_id using self.rules['source'].cat
        self.source_to_source_id = dict(zip(source_cat.categories, range(len(source_cat.categories))))

        self.matching_rules = []
        self.partials = []
        self.output = output
        self.ignore_own_page = ignore_own_page

    def handle_valid_change(self, prev_table, current_table, gen, version, correspondences):
        df_partials = pd.DataFrame(columns=['key', 'revisionId', 'source', 'target'], data=list(
            (version['key'], version['revisionId'], source, target) for source, target in gen))
        self.partials.append(df_partials)

        data = []
        for length in range(1, len(prev_table.header) + 1):
            for start in range(len(prev_table.header) - length + 1):
                seq = prev_table.header[start:start + length]

                # no matching rule
                if seq not in self.source_to_source_id:
                    continue

                data.append(
                    (self.source_to_source_id[seq], get_required_target(prev_table, current_table, start, length)))

        matching_rules = pd.DataFrame(columns=['source_id', 'required_target'], data=data)

        matching_rules['key'] = version['key']
        matching_rules['pageTitle'] = version['pageTitle']
        matching_rules['revisionId'] = version['revisionId']
        matching_rules['validFrom'] = dateutil.parser.parse(version['validFrom']).timestamp()

        self.matching_rules.append(matching_rules)
        return True

    def handle_complete(self):
        with catchtime_local('Concatenating partials'):
            df_partials = pd.concat(self.partials)
            df_partials.set_index(['key', 'revisionId', 'source', 'target'], inplace=True)
            del self.partials

        with catchtime_local('Concatenating matching rules'):
            df = pd.concat(self.matching_rules)
            del self.matching_rules
            df.set_index('source_id', inplace=True)
            df.sort_index(inplace=True)

        with catchtime_local('Calculating join with rules'):
            df = df.join(self.rules, how='inner')
            del self.rules

        logging.info("Size before excluding single-revision/key: {}".format(len(df)))
        with catchtime_local('Excluding single-revision'):
            df = df[~((df['single-revision'] == df['revisionId']) & (df['single-key'] == df['key']))]
            del df['single-revision']
            del df['single-key']
        logging.info("Size after excluding single-revision/key: {}".format(len(df)))

        df.reset_index(inplace=True)

        with catchtime_local('Calculating join with partials'):
            partial_idx = df.join(df_partials, on=['key', 'revisionId', 'source', 'target'], how='inner').index
            df['partial'] = False
            df.loc[partial_idx, 'partial'] = True
            df['full'] = df['required_target'] == df['target']

        # add distance information
        if self.embedding_model:
            with catchtime_local('Calculating distances'):
                self.add_distances(df)
            del self.embedding_model
            df.drop(columns=['pages'], inplace=True)

        # add date information
        with catchtime_local('Calculating date information'):
            df['closest-timediff'] = get_minimum_timediff(df['validFrom'], df['revisions'])
            df.drop(columns=['revisions', 'validFrom'], inplace=True)

        # add derived fields
        with catchtime_local('Calculating derived fields'):
            df['confidence_total'] = df['weight'] / df['weight_source']
            df['confidence_survived'] = df['survived'] / df['survived_source']
            df['popularity_ratio'] = df['weight_target'] / df['weight_source']
            df['popularity_ratio_survived'] = df['survived_target'] / df['survived_source']
            df['survivor_ratio'] = df['survived'] / df['weight']

        with catchtime_local('Calculating prepost fields'):
            df['precondition_length'] = df['source'].apply(lambda x: len(x))
            df['postcondition_length'] = df['target'].apply(lambda x: len(x))
            df['precondition'] = df['source'].apply(
                lambda x: ','.join(s[0] if s[0] else '' for s in x) if len(x) > 0 else None)
            df['postcondition'] = df['target'].apply(
                lambda x: ','.join(s[0] if s[0] else '' for s in x) if len(x) > 0 else None)

        with catchtime_local('Calculating score'):
            # if full match, score is 2
            df['score'] = 0
            # set the score to 1 where partial is true
            df.loc[df['partial'], 'score'] = 1
            # set the score to 2 where full is true
            df.loc[df['full'], 'score'] = 2

        # print memory usage
        logging.info(df.info(memory_usage='deep'))
        # print memory usage for each column
        logging.info(df.memory_usage(deep=True))

        with catchtime_local('Turn into categorial features'):
            df['precondition'] = df['precondition'].astype('category')
            df['postcondition'] = df['postcondition'].astype('category')
            df['pageTitle'] = df['pageTitle'].astype('category')
            df['key'] = df['key'].astype('category')
            logging.info(df.info(memory_usage='deep'))

        df = df[df.columns.difference(['source', 'target', 'full', 'partial', 'source_id', 'required_target'])]

        df.to_pickle(self.output)

    def add_distances(self, df):
        from rust_dist import get_distances_chunks
        df_with_pages = df.loc[~df['pages'].isnull()]
        if self.ignore_own_page:
            # remove pageTitle from pages
            df_with_pages['pages'] = df_with_pages.apply(
                lambda x: [page for page, count in x['pages'] if page != x['pageTitle'] or count > 1], axis=1)
        else:
            df_with_pages['pages'] = df_with_pages['pages'].apply(lambda x: [page for page, count in x])
        vector_dict = {page[7:].replace('_', ' '):
                           np.array(self.embedding_model.vectors[idx]) for page, idx in
                       self.embedding_model.key_to_index.items()}
        distances = get_distances_chunks(df_with_pages['pageTitle'].to_list(), df_with_pages['pages'].to_list(),
                                         vector_dict, 1024)
        df.loc[df_with_pages.index, ['euclidean_top1', 'euclidean_top5', 'cosine_top1', 'cosine_top5']] = distances
