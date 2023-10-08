import argparse
import fileinput
import itertools
import json
from collections import defaultdict

import dateutil.parser
import jsonlines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infobox', dest='infobox', action='store_true')
    parser.add_argument('--usefirstrow', dest='usefirstrow', action='store_true')
    parser.add_argument('--ignoreempty', dest='ignoreempty', action='store_true')
    parser.add_argument('--useheaderannotation', dest='useheaderannotation', action='store_true')
    parser.add_argument('files', metavar='FILE', nargs='*', help='files to read, if empty, stdin is used')

    args = parser.parse_args()

    for obj in read_and_group_input(args.files):
        handle_versions(obj, args.ignoreempty, args.infobox, args.usefirstrow, args.useheaderannotation)
        print(json.dumps(obj))


def read_and_group_input(files):
    schema_by_key = defaultdict(list)
    with jsonlines.Reader(fileinput.input(files=files if len(files) > 0 else ('-',))) as reader:
        for obj in reader:
            key = obj['key']
            schema_by_key[key].append(obj)
    for object_versions in schema_by_key.values():
        yield object_versions


def handle_versions(versions, ignoreempty, infobox, usefirstrow, useheaderannotation):
    for obj in versions:

        obj.pop('changes', None)
        obj.pop('attributes', None)
        if infobox and 'attributes' in obj:
            if ignoreempty:
                obj['header'] = []
                for key, value in obj['attributes'].items():
                    if value and value.strip():
                        obj['header'].append(key)
            else:
                obj['header'] = list(obj['attributes'].keys())
        if usefirstrow and 'contentParsed' in obj and len(obj['contentParsed']) > 0:
            obj['header'] = obj['contentParsed'][0]
        if useheaderannotation and 'contentParsed' in obj and len(obj['contentParsed']) > 0:
            header = []
            for row, props in zip(obj['contentParsed'], obj['contentProperties']):
                if len(row) < 1:
                    continue
                if get_header_amount(props) == 1.0 and not is_colspan_row(props):
                    header += [row]
                elif len(header) > 0:
                    break
            if len(header) == 0:
                obj['header'] = obj['contentParsed'][0]
            else:
                schema = [' '.join([x for (x, _) in itertools.groupby(a) if len(x) > 0]) for a in
                          itertools.zip_longest(*header, fillvalue='')]

                obj['header'] = schema
    add_previous_versions(versions)
    filtered = filter_same_schema(versions)
    filtered = filter_short_lived(filtered)
    filtered = filter_same_schema(filtered)
    return filtered


def get_header_amount(row):
    if len(row) == 0:
        return 0.0

    count = 0
    for c in row:
        if 'header' in c and c['header'] == True:
            count += 1
    header_share = count / len(row)
    return header_share


def is_colspan_row(row):
    if len(row) == 0:
        return False

    return 'colspan' in row[0] and row[0]['colspan'] >= len(row)


def add_previous_versions(versions):
    stable = versions[0]
    for i in range(1, len(versions)):
        obj = versions[i]
        obj['previous'] = dict(versions[i - 1])
        obj['previous'].pop('previous', None)
        obj['previous'].pop('previous-stable', None)
        obj['previous-stable'] = dict(stable)
        obj['previous-stable'].pop('previous', None)
        obj['previous-stable'].pop('previous-stable', None)
        if (i == len(versions) - 1) or is_stable(obj, versions[i + 1]):
            stable = obj


def filter_short_lived(versions):
    final = []
    for i in range(0, len(versions) - 1):
        obj = versions[i]
        if is_stable(obj, versions[i + 1]):
            final.append(obj)
    final.append(versions[-1])
    return final


def is_stable(current_version, next_version):
    duration = dateutil.parser.parse(next_version['validFrom']) - dateutil.parser.parse(current_version['validFrom'])
    return duration.days > 1 and ('header' not in current_version or all(len(s) < 100 for s in current_version['header']))


def filter_same_schema(versions):
    filtered = [versions[0]]
    for i in range(1, len(versions)):
        previous = filtered[-1]
        obj = versions[i]

        different_header = ('header' in obj) != ('header' in previous)
        header_in_both = ('header' in obj) and ('header' in previous)

        if different_header or (header_in_both and obj['header'] != previous['header']):
            filtered.append(obj)
    return filtered


if __name__ == '__main__':
    main()
