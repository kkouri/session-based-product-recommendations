import json
from pathlib import Path
import polars as pl

from src.data_generation.testset_labels import ground_truth, split_events, split_test_set

class TestGroundTruth:

    def test_ground_truth(self):
        eventimestamp = [{
            'itemid': 1000000,
            'timestamp': 1000000000001,
            'event': 'view'
        }, {
            'itemid': 1000001,
            'timestamp': 1000000000002,
            'event': 'view'
        }, {
            'itemid': 1000002,
            'timestamp': 1000000000003,
            'event': 'view'
        }, {
            'itemid': 1000003,
            'timestamp': 1000000000004,
            'event': 'view'
        }, {
            'itemid': 1000000,
            'timestamp': 1000000000005,
            'event': 'addtocart'
        }, {
            'itemid': 1000004,
            'timestamp': 1000000000006,
            'event': 'view'
        }, {
            'itemid': 1000004,
            'timestamp': 1000000000007,
            'event': 'addtocart'
        }, {
            'itemid': 1000005,
            'timestamp': 1000000000008,
            'event': 'view'
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000009,
            'event': 'view'
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000010,
            'event': 'addtocart'
        }, {
            'itemid': 100005,
            'timestamp': 1000000000011,
            'event': 'addtocart'
        }, {
            'itemid': 1000000,
            'timestamp': 1000000000012,
            'event': 'transaction'
        }, {
            'itemid': 1000004,
            'timestamp': 1000000000012,
            'event': 'transaction'
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000013,
            'event': 'view'
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000014,
            'event': 'addtocart'
        }]

        expected = [{
            'itemid': 1000000,
            'timestamp': 1000000000001,
            'event': 'view',
            'labels': {
                'addtocart': {1000000, 1000004, 1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000001,
            'timestamp': 1000000000002,
            'event': 'view',
            'labels': {
                'addtocart': {1000000, 1000004, 1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000002,
            'timestamp': 1000000000003,
            'event': 'view',
            'labels': {
                'addtocart': {1000000, 1000004, 1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000003,
            'timestamp': 1000000000004,
            'event': 'view',
            'labels': {
                'addtocart': {1000000, 1000004, 1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000000,
            'timestamp': 1000000000005,
            'event': 'addtocart',
            'labels': {
                'addtocart': {1000004, 1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000004,
            'timestamp': 1000000000006,
            'event': 'view',
            'labels': {
                'addtocart': {1000004, 1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000004,
            'timestamp': 1000000000007,
            'event': 'addtocart',
            'labels': {
                'addtocart': {1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000005,
            'timestamp': 1000000000008,
            'event': 'view',
            'labels': {
                'addtocart': {1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000009,
            'event': 'view',
            'labels': {
                'addtocart': {1000007, 100005},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000010,
            'event': 'addtocart',
            'labels': {
                'addtocart': {100005, 1000007},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 100005,
            'timestamp': 1000000000011,
            'event': 'addtocart',
            'labels': {
                'addtocart': {1000007},
                'transaction': {1000000, 1000004}
            }
        }, {
            'itemid': 1000000,
            'timestamp': 1000000000012,
            'event': 'transaction',
            'labels': {
                'addtocart': {1000007},
                'transaction': {1000004}
            }
        }, {
            'itemid': 1000004,
            'timestamp': 1000000000012,
            'event': 'transaction',
            'labels': {
                'addtocart': {1000007}
            }
        }, {
            'itemid': 1000007,
            'timestamp': 1000000000013,
            'event': 'view',
            'labels': {
                'addtocart': {1000007},
            }
        }]

        result = ground_truth(eventimestamp)
        assert result == expected


class TestSplitEvents:
    def test_split_events_short(self):
        events_short = [{
            'itemid': 1,
            'timestamp': 1000000000001,
            'event': 'view'
        }, {
            'itemid': 2,
            'timestamp': 1000000000002,
            'event': 'addtocart'
        }]

        result = split_events(events_short)

        expected_events = [{'itemid': 1, 'timestamp': 1000000000001, 'event': 'view'}]
        expected_labels = {'addtocart': {2}}

        assert result[0] == expected_events
        assert result[1] == expected_labels

    def test_split_events_long(self):
        events_long = [{
            'itemid': 1,
            'timestamp': 1000000000001,
            'event': 'view'
        }, {
            'itemid': 1,
            'timestamp': 1000000000002,
            'event': 'addtocart'
        }, {
            'itemid': 1,
            'timestamp': 1000000000003,
            'event': 'transaction'
        }, {
            'itemid': 2,
            'timestamp': 1000000000004,
            'event': 'view'
        }]

        result = split_events(events_long, 1)
        expected_events = [{'itemid': 1, 'timestamp': 1000000000001, 'event': 'view'}]
        expected_labels = {'addtocart': {1}, 'transaction': {1}}

        assert result[0] == expected_events
        assert result[1] == expected_labels

        result = split_events(events_long, 2)
        expected_events = [{
            'itemid': 1,
            'timestamp': 1000000000001,
            'event': 'view'
        }, {
            'itemid': 1,
            'timestamp': 1000000000002,
            'event': 'addtocart'
        }]
        expected_labels = {'transaction': {1}}
        assert result[0] == expected_events
        assert result[1] == expected_labels
