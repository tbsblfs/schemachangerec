import snapshottest

from schemarec.mining.change_projection import execute_strategy, get_sources_and_targets, \
    get_positions


def generate_subsequences(header):
    yield 0,0
    for i in range(len(header)):
        for j in range(i + 1, len(header)+1):
            yield i, j

def generate_all_combinations(previous_header, current_header, correspondences, require_maximal=False):
    sources, targets = get_sources_and_targets(current_header, previous_header, correspondences)
    # for each field in the previous header get their position in the current header
    target_positions = get_positions(current_header, previous_header, targets)
    # for each field in the current header get their position in the previous header
    source_positions = get_positions(previous_header, current_header, sources)
    for i, j in generate_subsequences(previous_header):
        if i == j:
            continue
        for k, l in generate_subsequences(current_header):
            if not all((s is None) or (i <= s < j) for s in source_positions[k:l]):
                continue
            if not all((t is None) or (k <= t < l) for t in target_positions[i:j]):
                continue
            if k != l and all(s is None for s in source_positions[k:l]):
                continue
            if require_maximal:
                if k != l and k > 0 and source_positions[k-1] is None:
                    continue
                if k != l and l < len(current_header) and source_positions[l] is None:
                    continue
                #if i > 0 and target_positions[i-1] is None:
                #    continue
                #if j < len(previous_header) and target_positions[j] is None:
                #    continue

            yield previous_header[i:j], current_header[k:l]


class ChangeProjectionTestCase(snapshottest.TestCase):
    def test_stategy_execution(self):
        previous_header = ('a', 'b', 'c', 'd', 'e')
        current_header = ('f', 'g', 'h', 'i', 'j')

        correspondences = [('i', 'b'), ('j', 'd'), ('g', 'e')]
        projections = list(execute_strategy(correspondences, current_header, previous_header, 'include'))
        self.assertMatchSnapshot(projections, 'projections_include')


    def test_strategy_execution(self):
        previous_header = ('a', 'b', 'c', 'd', 'e')
        current_header = ('f', 'g', 'h', 'i', 'j')

        correspondences = [('i', 'b'), ('j', 'd'), ('g', 'e')]

        mapping_exclude = [s for s  in execute_strategy(correspondences, current_header, previous_header, 'exclude')]
        mapping_all = [s for s in generate_all_combinations(previous_header, current_header, correspondences, require_maximal=False)]

        self.assertEqual(set(mapping_exclude), set(mapping_all))

    def test_symmetry(self):
        previous_header = ('a', 'b', 'c', 'd', 'e')
        current_header = ('f', 'g', 'h', 'i', 'j')

        correspondences = [('i', 'b'), ('j', 'd'), ('g', 'e')]

        correspondences_symmetric = [(b, a) for a, b in correspondences]

        mappings_1 = [s for s  in execute_strategy(correspondences, current_header, previous_header, 'exclude')]
        mappings_inverse = [(b, a) for a,b  in execute_strategy(correspondences_symmetric, previous_header, current_header, 'exclude')]

        self.assertMatchSnapshot(set(mappings_1) - set(mappings_inverse), 'additional_mappings')
        self.assertMatchSnapshot(set(mappings_inverse) - set(mappings_1), 'missing_mappings')

    def test_symmetry2(self):
        previous_header = ('a', 'b', 'c', 'd', 'e')
        current_header = ('f', 'g', 'h', 'i', 'j')

        correspondences = [('i', 'b'), ('j', 'd'), ('g', 'e')]

        correspondences_symmetric = [(b, a) for a, b in correspondences]

        mappings_1 = [s for s in generate_all_combinations(previous_header, current_header, correspondences, require_maximal=False)]
        mappings_inverse = [(b, a) for a,b in generate_all_combinations(current_header, previous_header, correspondences_symmetric, require_maximal=False)]

        self.assertMatchSnapshot(set(mappings_1) - set(mappings_inverse), 'additional_mappings2')
        self.assertMatchSnapshot(set(mappings_inverse) - set(mappings_1), 'missing_mappings2')


