from src.evaluate import (evaluate_session, evaluate_sessions, get_scores, num_events, recall_by_event_type, mrr_by_event_type)


class TestEvaluate:

    def test_evaluate_session(self):
        label_last_event = {
            'addtocart': {1000000, 1000004, 1000007, 100005},
            'transaction': {1000000, 1000004}
        }
        prediction = {
            'addtocart': [1000004, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            'transaction': [1000000, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        }
        expected_hits = {'addtocart': 1, 'transaction': 1}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=20)

        label_last_event = {'addtocart': {1000000, 1000004, 1000007, 100005}, 'transaction': set()}
        prediction = {
            'addtocart': [1000004, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            'transaction': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        expected_hits = {'addtocart': 1, 'transaction': None}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=20)

        label_last_event = {'transaction': {1000000, 1000004}}
        prediction = {
            'addtocart': [1000004, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            'transaction': [1000000, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        }
        expected_hits = {'addtocart': None, 'transaction': 1}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=20)

        label_last_event = {'transaction': {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}
        prediction = {
            'addtocart': [1000004, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            'transaction': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        expected_hits = {'addtocart': None, 'transaction': 10}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=10)

        label_last_event = {'addtocart': set(), 'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}}
        prediction = {
            'addtocart': [1000004, 0, 1, 2, 3, 4, 5, 6, 7, 8],
            'transaction': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        }
        expected_hits = {'addtocart': None, 'transaction': 6}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=10)

        label_last_event = {
            'addtocart': {5, 6, 7, 8, 9},
            'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}
        }
        prediction = {'addtocart': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
        expected_hits = {'addtocart': 5, 'transaction': None}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=10)

        label_last_event = {
            'addtocart': {5, 6, 7, 8, 9},
            'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}
        }
        prediction = {'addtocart': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'transaction': None}
        expected_hits = {'addtocart': 5, 'transaction': None}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=10)

        label_last_event = {
            'addtocart': {5, 6, 7, 8, 9},
            'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}
        }
        prediction = {}
        expected_hits = {'addtocart': None, 'transaction': None}
        assert expected_hits == evaluate_session(label_last_event, prediction, k=10)

    def test_evaluate_sessions(self):
        k = 3
        predictions = {
            1: {
                'addtocart': [1, 2, 3],
                'transaction': [1, 32, 33]
            },
            2: {
                'addtocart': [1000004, 2, 3, 1000007],
                'transaction': [1, 2, 3]
            },
            3: {
                'addtocart': [1000007, 1000004, 3],
                'transaction': [1, 2, 3]
            }
        }
        labels = {
            1: {
                'addtocart': set(),
                'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}
            },
            2: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': {1000000, 1000004}
            },
            3: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': set()
            },
            4: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': set()
            }
        }

        expected = {
            1: {
                'addtocart': None,
                'transaction': 1
            },
            2: {
                'addtocart': 1,
                'transaction': 0
            },
            3: {
                'addtocart': 2,
                'transaction': None
            },
            4: {
                'addtocart': 0,
                'transaction': None
            }
        }

        assert expected == evaluate_sessions(labels, predictions, k)

    def test_num_events(self):
        k = 10
        labels = {
            1: {
                'addtocart': set(),
                'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}
            },
            2: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': {1000000, 1000004}
            },
            3: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': set()
            },
            4: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': set()
            }
        }

        expected = {'addtocart': 9, 'transaction': 12}

        assert expected == num_events(labels, k)

    def test_recall_by_event_type(self):

        total_number_events = {'addtocart': 10, 'transaction': 4}

        elementwise_evaluation = {
            1: {
                'addtocart': None,
                'transaction': 1
            },
            2: {
                'addtocart': 1,
                'transaction': 0
            },
            3: {
                'addtocart': 2,
                'transaction': None
            },
            4: {
                'addtocart': 0,
                'transaction': None
            }
        }

        expected_recall = {
            'addtocart': (1 + 2 + 0) / total_number_events['addtocart'],
            'transaction': (1 + 0) / total_number_events['transaction']
        }

        assert expected_recall == recall_by_event_type(elementwise_evaluation, total_number_events)

    def test_mrr_by_event_type(self):
        predictions = {
            1: {
                'addtocart': [1, 2, 3],
                'transaction': [1, 2, 3]
            }
        }

        labels = {
            1: {
                'addtocart': {1},
                'transaction': {0, 1}
            }
        }

        # First hit
        assert mrr_by_event_type(predictions, labels) == {'addtocart': 1.0, 'transaction': 1.0}

        labels = {
            1: {
                'addtocart': {4},
                'transaction': {4}
            }
        }

        # No hit
        assert mrr_by_event_type(predictions, labels) == {'addtocart': 0.0, 'transaction': 0.0}

        labels = {
            1: {
                'transaction': set()
            }
        }

        # No label for addtocart or transaction
        assert mrr_by_event_type(predictions, labels) == {'addtocart': None, 'transaction': None}

        predictions = {
            1: {
                'addtocart': [1, 2, 3],
                'transaction': [1, 2, 3]
            },
            2: {
                'addtocart': [1, 2, 3],
                'transaction': [1, 2, 3]
            },
            3: {
                'addtocart': [1, 2, 3],
                'transaction': [1, 2, 3]
            }
        }

        labels = {
            1: {
                'addtocart': {1},
                'transaction': {0, 1}
            },
            2: {
                'addtocart': {2, 3},
            },
            3: {
                'addtocart': {10},
                'transaction': {10}
            }
        }

        # Different hits
        assert mrr_by_event_type(predictions, labels) == {'addtocart': 0.5, 'transaction': 0.5}

    def test_get_scores(self):
        k = 3
        predictions = {
            1: {
                'addtocart': [1, 2, 3],
                'transaction': [1, 32, 33]
            },
            2: {
                'addtocart': [1000004, 2, 3, 1000007],
                'transaction': [1, 2, 3]
            },
            3: {
                'addtocart': [1000007, 1000004, 3],
                'transaction': [1, 2, 3]
            }
        }
        labels = {
            1: {
                'addtocart': set(),
                'transaction': {0, 1, 2, 3, 4, 5, 16, 17, 18, 19, 20}
            },
            2: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': {1000000, 1000004}
            },
            3: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': set()
            },
            4: {
                'addtocart': {1000000, 1000004, 1000007},
                'transaction': set()
            }
        }

        expected_evaluated_events = {
            1: {
                'addtocart': None,
                'transaction': 1
            },
            2: {
                'addtocart': 1,
                'transaction': 0
            },
            3: {
                'addtocart': 2,
                'transaction': None
            },
            4: {
                'addtocart': 0,
                'transaction': None
            }
        }

        expected_scores = ({
            'addtocart':
            3 / 9,
            'transaction':
            1 / 5
        },
            {'addtocart': 1.0, 'transaction': 0.5}
        )

        assert expected_evaluated_events == evaluate_sessions(labels, predictions, k)
        assert expected_scores == get_scores(labels, predictions, k)
