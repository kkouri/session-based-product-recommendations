import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import polars as pl
from beartype import beartype
from tqdm.auto import tqdm


class setEncoder(json.JSONEncoder):

    def default(self, obj):
        return list(obj)


@beartype
def ground_truth(events: list[dict]):
    prev_labels = {"addtocart": set(), "transaction": set()}

    for event in reversed(events):
        event["labels"] = {}

        for label in ['addtocart', 'transaction']:
            if prev_labels[label]:
                event["labels"][label] = prev_labels[label].copy()

        if event["event"] == "addtocart":
            prev_labels['addtocart'].add(event["itemid"])
        elif event["event"] == "transaction":
            prev_labels['transaction'].add(event["itemid"])

    return events[:-1]


@beartype
def split_events(events: list[dict], split_idx: int | None = None):
    test_events = ground_truth(deepcopy(events))
    # Make sure that there will be addtocart or transaction event in the test set
    if split_idx is None:
        last_possible_idx = 0
        for i, event in enumerate(test_events):
            if event['labels']:
                last_possible_idx = i
        split_idx = 1
        if last_possible_idx != 0:
            split_idx = random.randint(1, last_possible_idx)

    test_events = test_events[:split_idx]
    labels = test_events[-1]['labels']
    for event in test_events:
        del event['labels']
    return test_events, labels


@beartype
def split_test_set(sessions: pl.DataFrame, sessions_output: Path, labels_output: Path):
    last_labels = []
    splitted_sessions = []

    sessions = sessions.rows()
    for session_id, events in tqdm(sessions, desc="Creating trimmed testset", total=len(sessions)):
        if len(events) < 2:
            continue
        splitted_events, labels = split_events(events)
        last_labels.append({'session': session_id, 'labels': labels})
        splitted_sessions.append({'session': session_id, 'events': splitted_events})

    with open(sessions_output, 'w') as f:
        for session in splitted_sessions:
            f.write(json.dumps(session) + '\n')

    with open(labels_output, 'w') as f:
        for label in last_labels:
            f.write(json.dumps(label, cls=setEncoder) + '\n')


@beartype
def main(test_set: Path, output_path: Path, seed: int):
    random.seed(seed)
    # read test set and squeeze session events into a single row
    test_sessions = (
        pl.read_csv(test_set)
        .sort(["session", "timestamp"])
        .select("session", pl.struct("itemid", "timestamp", "event").alias("events"))
        .group_by("session")
        .agg(pl.col("events"))
    )
    test_sessions_file = output_path / 'test_sessions.jsonl'
    test_labels_file = output_path / 'test_labels.jsonl'
    split_test_set(test_sessions, test_sessions_file, test_labels_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-set', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.test_set, args.output_path, args.seed)
