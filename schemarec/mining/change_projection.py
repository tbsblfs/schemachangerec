def execute_strategy(correspondences, current_header, prev_header, strategy):
    if strategy == 'whole':
        if prev_header != current_header:
            yield tuple(prev_header), tuple(current_header)
        return
    sources, targets = get_sources_and_targets(current_header, prev_header, correspondences)
    target_positions = get_positions(current_header, prev_header, targets)
    source_positions = get_positions(prev_header, current_header, sources)
    for length in range(1, len(prev_header) + 1):
        for start in range(len(prev_header) - length + 1):
            if all(s is None for s in target_positions[start:start + length]):
                yield tuple(prev_header[start:start + length]), tuple()
                continue

            min_target, max_target, min_target_extended, max_target_extended = find_extent(length, source_positions,
                                                                                           start, target_positions)
            for target_start in range(min_target_extended, min_target + 1):
                for target_end in range(max_target, max_target_extended + 1):
                    source_schema = prev_header[start:start + length]
                    target_schema = current_header[target_start:target_end + 1]

                    in_range_sources = [s is None or start <= s < start + length for s in
                                        source_positions[target_start:target_end + 1]]

                    for target_schema in handle_reordering(strategy, in_range_sources, target_schema):
                        if source_schema != target_schema:
                            # add edge to G or increment weight if it already exists
                            yield tuple(source_schema), tuple(target_schema)


def seq_split(lst, cond):
    sublist = []
    for pos, item in enumerate(lst):
        if cond(pos):
            sublist.append(item)
        elif sublist:
            yield sublist
            sublist = []
    if sublist:
        yield sublist


def get_sources_and_targets(current_header, prev_header, correspondences):
    sources = {s: s for s in current_header}
    targets = {s: s for s in prev_header}
    for t, s in correspondences:
        targets[s[0]] = t[0]
        sources[t[0]] = s[0]
    return sources, targets


def get_positions(current_header, prev_header, targets):
    positions = {c: i for i, c in enumerate(current_header)}
    return [positions[targets[c]] if targets[c] in positions else None
            for c in prev_header]


def handle_reordering(reordering_strategy, in_range_sources, target_schema):
    if reordering_strategy == 'include':
        yield target_schema
    elif reordering_strategy == 'exclude':
        if all(in_range_sources):
            yield target_schema
        return
    elif reordering_strategy == 'project':
        yield [t for t, in_range in zip(target_schema, in_range_sources) if in_range]
    elif reordering_strategy == 'split':
        for s in seq_split(target_schema, lambda pos: in_range_sources[pos]):
            yield s


def find_extent(length, source_positions, start, target_positions):
    min_target = min(s for s in target_positions[start:start + length] if s is not None)
    # extend to the left
    min_target_original = min_target
    while min_target > 0 and source_positions[min_target - 1] is None:
        min_target -= 1
    max_target = max(s for s in target_positions[start:start + length] if s is not None)
    max_target_original = max_target
    # extend to the right
    while max_target < len(source_positions) - 1 and source_positions[max_target + 1] is None:
        max_target += 1
    return min_target_original, max_target_original, min_target, max_target

