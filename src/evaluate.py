import argparse
import json
import logging
from pathlib import Path

from beartype import beartype
from tqdm.auto import tqdm


@beartype
def prepare_predictions(predictions: list[str]):
    prepared_predictions = dict()
    for prediction in tqdm(predictions, desc="Preparing predictions"):
        sid_type, preds = prediction.strip().split(",")
        sid, event_type = sid_type.split("_")
        preds = [int(aid) for aid in preds.split(" ")] if preds != "" else []
        if not int(sid) in prepared_predictions:
            prepared_predictions[int(sid)] = dict()
        prepared_predictions[int(sid)][event_type] = preds
    return prepared_predictions


@beartype
def prepare_labels(labels: list[str]):
    final_labels = dict()
    for label in tqdm(labels, desc="Preparing labels"):
        label = json.loads(label)
        final_labels[label["session"]] = {
            "addtocart": set(label["labels"].get("addtocart", [])),
            "transaction": set(label["labels"].get("transaction", []))
        }
    return final_labels


@beartype
def evaluate_session(labels: dict, prediction: dict, k: int):
    if 'addtocart' in labels and labels['addtocart'] and 'addtocart' in prediction and prediction['addtocart']:
        cart_hits = len(set(prediction['addtocart'][:k]).intersection(labels['addtocart']))
    else:
        cart_hits = None

    if 'transaction' in labels and labels['transaction'] and 'transaction' in prediction and prediction['transaction']:
        order_hits = len(set(prediction['transaction'][:k]).intersection(labels['transaction']))
    else:
        order_hits = None

    return {'addtocart': cart_hits, 'transaction': order_hits}


@beartype
def evaluate_sessions(labels: dict[str | int, dict], predictions: dict[int, dict], k: int):
    result = {}
    for session_id, session_labels in tqdm(labels.items(), desc="Evaluating sessions"):
        if session_id in predictions:
            result[session_id] = evaluate_session(session_labels, predictions[session_id], k)
        else:
            result[session_id] = {k: 0. if v else None for k, v in session_labels.items()}
    return result


@beartype
def num_events(labels: dict[int, dict], k: int):
    num_carts = 0
    num_orders = 0
    for event in labels.values():
        if 'addtocart' in event and event['addtocart']:
            num_carts += min(len(event['addtocart']), k)
        if 'transaction' in event and event['transaction']:
            num_orders += min(len(event['transaction']), k)
    return {'addtocart': num_carts, 'transaction': num_orders}


@beartype
def recall_by_event_type(evalutated_events: dict, total_number_events: dict):
    carts = 0
    orders = 0
    for event in evalutated_events.values():
        if 'addtocart' in event and event['addtocart']:
            carts += event['addtocart']
        if 'transaction' in event and event['transaction']:
            orders += event['transaction']

    return {
        'addtocart': carts / total_number_events['addtocart'],
        'transaction': orders / total_number_events['transaction']
    }


@beartype
def mrr_by_event_type(predictions: dict, labels: dict):
    carts_ranks = []
    transactions_ranks = []

    for session in predictions.keys():
        session_predictions = predictions[session]

        if 'addtocart' in session_predictions and 'addtocart' in labels[session] and labels[session]['addtocart']:
            reciprocal_rank = 0
            for i, itemid in enumerate(session_predictions['addtocart'], start=1):
                if itemid in labels[session]['addtocart']:
                    reciprocal_rank = 1 / i
                    break
            carts_ranks.append(reciprocal_rank)

        if 'transaction' in session_predictions and 'transaction' in labels[session] and labels[session]['transaction']:
            reciprocal_rank = 0
            for i, itemid in enumerate(session_predictions['transaction'], start=1):
                if itemid in labels[session]['transaction']:
                    reciprocal_rank = 1 / i
                    break
            transactions_ranks.append(reciprocal_rank)

    return {
        'addtocart': sum(carts_ranks) / len(carts_ranks) if carts_ranks else None,
        'transaction': sum(transactions_ranks) / len(transactions_ranks) if transactions_ranks else None
    }


@beartype
def get_scores(labels: dict[int, dict], predictions: dict[int, dict], k=20):
    '''
    Calculates the recall and mrr for the given predictions and labels.
    Args:
        labels: dict of labels for each session
        predictions: dict of predictions for each session
        k: cutoff for the recall calculation
    Returns:
        recalls for each event type
        mrrs for each event type
    '''
    total_number_events = num_events(labels, k)
    evaluated_events = evaluate_sessions(labels, predictions, k)
    recalls = recall_by_event_type(evaluated_events, total_number_events)
    mrrs = mrr_by_event_type(predictions, labels)
    return recalls, mrrs


@beartype
def main(labels_path: Path, predictions_path: Path):
    with open(labels_path, "r") as f:
        logging.info(f"Reading labels from {labels_path}")
        labels = f.readlines()
        labels = prepare_labels(labels)
        logging.info(f"Read {len(labels)} labels")
    with open(predictions_path, "r") as f:
        logging.info(f"Reading predictions from {predictions_path}")
        predictions = f.readlines()[1:]
        predictions = prepare_predictions(predictions)
        logging.info(f"Read {len(predictions)} predictions")
    logging.info("Calculating scores")
    k = 20
    recalls, mrrs = get_scores(labels, predictions, k)
    logging.info(f"Recall@{k} scores: {recalls}")
    logging.info(f"MRR@{k} scores: {recalls}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-labels', default="resources/test_labels.jsonl", type=str)
    parser.add_argument('--predictions', default="resources/predictions.csv", type=str)
    args = parser.parse_args()
    main(Path(args.test_labels), Path(args.predictions))
