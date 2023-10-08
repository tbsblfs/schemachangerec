import fileinput
from pathlib import Path

import pandas as pd
import streamlit as st
from analysis.schemamatching.schema_matching import create_header
from jsonlines import jsonlines

input_path = Path("//data/column-gold/")


def main():
    st.set_page_config(layout="wide")
    data = load_data()
    gold = load_gold()
    finished = load_finished()
    finished_ids = set([s['revisionId'] for s in finished])

    group = st.selectbox("Groups", data.keys(), index=0)
    first_unfinished = next((i for i, s in enumerate(data[group]) if s['revisionId'] not in finished_ids), None)
    revision = st.selectbox("Revision", [o["revisionId"] for o in data[group]],
                            index=first_unfinished if first_unfinished is not None else len(data[group]) - 1)

    show_progress(data[group], finished_ids)
    obj = next(o for o in data[group] if o['revisionId'] == revision)

    gold_current_revision = [g for g in gold if g['revisionId'] == obj['revisionId']]

    used_added = set(tuple(x) for g in gold_current_revision for x in g['added'])
    used_removed = set(tuple(x) for g in gold_current_revision for x in g['removed'])

    current_header = create_header(obj)
    previous_header = create_header(obj['previous-stable'])
    added = list(set(current_header).difference(set(previous_header)) - used_added)
    added.sort(key=lambda x: x[0] or "")
    removed = list(set(previous_header).difference(set(current_header)) - used_removed)
    removed.sort(key=lambda x: x[0] or "")

    with st.form("correspondence_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            st.header("Previous")
            st.write(
                f"[{obj['previous-stable']['revisionId']}](https://en.wikipedia.org/?oldid={obj['previous-stable']['revisionId']})")
            st.dataframe(pd.DataFrame(obj['previous-stable']['contentParsed'], columns=previous_header if len(
                previous_header) > 0 else None), use_container_width=True)
            for rem in removed:
                st.checkbox(turn_to_string(rem), key=f"rem-{rem}")

        with col2:
            st.header("Current")
            st.write(f"[{obj['revisionId']}](https://en.wikipedia.org/?oldid={obj['revisionId']})")
            st.dataframe(pd.DataFrame(obj['contentParsed'], columns=current_header if len(
                current_header) > 0 else None), use_container_width=True)
            for add in added:
                st.checkbox(turn_to_string(add), key=f"add-{add}")

        st.form_submit_button("Add", on_click=save_gold, args=(obj, added, removed))

    with st.form("save_form", clear_on_submit=True):
        for g in gold_current_revision:
            st.write(','.join(map(turn_to_string, g['removed'])) + '->' + ','.join(map(turn_to_string, g['added'])))

        comment = st.text_input("Comment", key="comment",
                                value=next((s["comment"] for s in finished if s['revisionId'] == obj['revisionId']),
                                           ""))
        skipped = st.checkbox("Skip", key="skipped",
                              value=next((s["skipped"] for s in finished if s['revisionId'] == obj['revisionId']),
                                         False))
        st.form_submit_button("Save & Next", on_click=save_finished, args=(obj, comment, skipped))


def show_progress(data, finished_ids):
    todo = len([o for o in data if o['revisionId'] not in finished_ids])
    st.progress(1.0 - todo / len(data))


def turn_to_string(header):
    if header[1] == 0: return str(header[0])
    return f"{header[0]} ({header[1]})"


def save_gold(obj, added, removed):
    with open(input_path / "gold.json", 'a') as f:
        with jsonlines.Writer(f) as writer:
            writer.write({'added': [s for s in added if st.session_state[f"add-{s}"]],
                          'removed': [s for s in removed if st.session_state[f"rem-{s}"]],
                          'revisionId': obj['revisionId']})


def save_finished(obj):
    with open(input_path / "finished.json", 'a') as f:
        with jsonlines.Writer(f) as writer:
            writer.write({'revisionId': obj['revisionId'], 'comment': st.session_state.comment,
                          'skipped': st.session_state.skipped})


@st.experimental_memo
def load_data():
    data = {}
    with jsonlines.Reader(fileinput.input(files=input_path / "sample_strat.json")) as reader:
        for s in reader:
            data = s
    return data


def load_gold():
    with jsonlines.Reader(fileinput.input(files=input_path / "gold.json")) as reader:
        return [s for s in reader]


def load_finished():
    with jsonlines.Reader(fileinput.input(files=input_path / "finished.json")) as reader:
        return list(s for s in reader)


if __name__ == '__main__':
    main()
