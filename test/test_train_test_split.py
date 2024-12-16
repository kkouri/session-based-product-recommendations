from pathlib import Path
import polars as pl

from src.data_generation.train_test_split import create_sessions, create_train_test_split


class TestCreateSessions:

    def test_split_session_based_on_event_time_diff(self):
        """
        Time difference can be maximum of 30 minutes
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000100, 1000010000, 1000011000],
            "visitorid": [1, 1, 1, 1],
            "event": ["view", "addtocart", "transaction", "view"],
            "itemid": [1, 1, 1, 2, ],
            "transactionid": [None, None, 1, None, ]
        })

        result = create_sessions(events_df).to_dict(as_series=False)

        expected_sessions = {
            "timestamp": [1000000000, 1000000100, 1000010000, 1000011000],
            "event": ["view", "addtocart", "transaction", "view"],
            "itemid": [1, 1, 1, 2],
            "session": [1, 1, 2, 2]
        }

        assert result == expected_sessions

    def test_split_session_based_on_user(self):
        """
        If the user is different, the session should be different
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000100, 1000000000, 1000000100],
            "visitorid": [1, 1, 2, 2],
            "event": ["view", "addtocart", "view", "addtocart"],
            "itemid": [1, 1, 3, 3],
            "transactionid": [None, None, None, None]
        })

        result = create_sessions(events_df).to_dict(as_series=False)

        expected_sessions = {

            "timestamp": [1000000000, 1000000100, 1000000000, 1000000100],
            "event": ["view", "addtocart", "view", "addtocart"],
            "itemid": [1, 1, 3, 3],
            "session": [1, 1, 2, 2]
        }

        assert result == expected_sessions

    def test_remove_sessions_with_one_event(self):
        """
        Sessions with only one event should be removed
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000100, 1000010000],
            "visitorid": [1, 1, 1],
            "event": ["view", "addtocart", "view"],
            "itemid": [1, 1, 1],
            "transactionid": [None, None, None]
        })

        result = create_sessions(events_df).to_dict(as_series=False)

        expected_sessions = {
            "timestamp": [1000000000, 1000000100],
            "event": ["view", "addtocart"],
            "itemid": [1, 1],
            "session": [1, 1]
        }

        assert result == expected_sessions

    def test_all_conditions(self):
        """
        Test all conditions together
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000100, 1000010000, 1000011000, 1000000000, 1000000100, 1000000200,
                          1000010300, 1000000400, 1000000500],
            "visitorid": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            "event": ["view", "addtocart", "transaction", "view", "view", "addtocart", "addtocart", "view", "view",
                      "view"],
            "itemid": [1, 1, 1, 2, 3, 3, 3, 4, 5, 6],
            "transactionid": [None, None, 1, None, None, None, None, None, None, None]
        })

        result = create_sessions(events_df).to_dict(as_series=False)

        expected_sessions = {
            "timestamp": [1000000000, 1000000100, 1000010000, 1000011000, 1000000000, 1000000100, 1000000200,
                          1000000400,
                          1000000500],
            "event": ["view", "addtocart", "transaction", "view", "view", "addtocart", "addtocart", "view", "view"],
            "itemid": [1, 1, 1, 2, 3, 3, 3, 5, 6],
            "session": [1, 1, 2, 2, 3, 3, 3, 4, 4]
        }

        assert result == expected_sessions


class TestCreateTrainTestSplit:

    one_week_in_seconds = 604_800

    def test_split_based_on_weeks(self):
        """
        Train/test split based on weeks
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000000, 1000604801, 1000604900],
            "event": ["addtocart", "addtocart", "addtocart", "addtocart"],
            "itemid": [1, 1, 1, 1],
            "session": [1, 1, 2, 2]
        })

        result = create_train_test_split(events_df, 1, 1).to_dict(as_series=False)

        expected_train = {
            "timestamp": [1000000000, 1000000000],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "transactionid": [None, None],
            "session": [1, 1]
        }

        expected_test = {
            "timestamp": [1000604801, 1000604900],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "session": [2, 2]
        }

        assert result == (expected_train, expected_test)

    def test_filter_sessions_with_no_cart_or_transaction_events(self):
        """
        Filter out sessions with no cart or transaction events
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000000, 1000000000, 1000000000, 1000604801, 1000604900, 1000604801, 1000604900],
            "event": ["addtocart", "addtocart", "view", "view", "addtocart", "addtocart", "view", "view"],
            "itemid": [1, 1, 1, 1, 1, 1, 1, 1],
            "session": [1, 1, 2, 2, 3, 3, 4, 4]
        })

        result = create_train_test_split(events_df, 1, 1).to_dict(as_series=False)

        expected_train = {
            "timestamp": [1000000000, 1000000000],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "transactionid": [None, None],
            "session": [1, 1]
        }

        expected_test = {
            "timestamp": [1000604801, 1000604900],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "session": [2, 2]
        }

        assert result == (expected_train, expected_test)

    def test_filter_sessions_with_only_cart_or_transaction_as_first_event(self):
        """
        Filter out sessions with only cart or transaction as the first event. This is same as having the session with
        only views.
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000000, 1000000000, 1000000000, 1000604801, 1000604900, 1000604801,
                          1000604900],
            "event": ["addtocart", "addtocart", "addtocart", "view", "addtocart", "addtocart", "addtocart", "view"],
            "itemid": [1, 1, 1, 1, 1, 1, 1, 1],
            "session": [1, 1, 2, 2, 3, 3, 4, 4]
        })

        result = create_train_test_split(events_df, 1, 1).to_dict(as_series=False)

        expected_train = {
            "timestamp": [1000000000, 1000000000],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "transactionid": [None, None],
            "session": [1, 1]
        }

        expected_test = {
            "timestamp": [1000604801, 1000604900],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "session": [2, 2]
        }

        assert result == (expected_train, expected_test)

    def test_filter_out_train_sessions_with_length_of_one(self):
        """
        Filter out train sessions with length of one. This happens when the session is cut off by the train/test split
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000050, 1000000000, 1000000300, 1000604850, 1000604900],
            "event": ["addtocart", "addtocart", "addtocart", "addtocart", "addtocart", "addtocart"],
            "itemid": [1, 1, 1, 1, 1, 1],
            "session": [1, 1, 2, 2, 3, 3]
        })

        train_result, _ = create_train_test_split(events_df, 1, 1).to_dict(as_series=False)

        expected_train = {
            "timestamp": [1000000000, 1000000050],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "transactionid": [None, None],
            "session": [1, 1]
        }

        assert train_result == expected_train

    def test_filter_out_test_sessions_that_appear_in_train_dataset(self):
        """
        Filter out test sessions that appear in the train dataset
        """

        events_df = pl.DataFrame({
            "timestamp": [1000000000, 1000000050, 1000000000, 1000000050, 1000000300, 1000000350, 1000604850, 1000604900],
            "event": ["addtocart", "addtocart", "addtocart", "addtocart", "addtocart", "addtocart", "addtocart", "addtocart"],
            "itemid": [1, 1, 1, 1, 1, 1, 1, 1],
            "session": [1, 1, 2, 2, 2, 2, 3, 3]
        })

        _, test_result = create_train_test_split(events_df, 1, 1).to_dict(as_series=False)

        expected_test = {
            "timestamp": [1000604850, 1000604900],
            "event": ["addtocart", "addtocart"],
            "itemid": [1, 1],
            "session": [3, 3]
        }

        assert test_result == expected_test

    def test_train_test_timestamp_overlap(self):
        """
        No timestamps overlap in train and test datasets
        """

        events_df = (
            pl.read_csv(Path("../data/events.csv"))
            # Convert timestamp to seconds and cast to UInt32 to save memory
            .with_columns((pl.col("timestamp")//1000).cast(pl.UInt32))
        )
        sessions_df = create_sessions(events_df)
        train_df, test_df = create_train_test_split(sessions_df, 1, 1)

        train_max_ts = train_df.select("timestamp").max().item()
        test_min_ts = test_df.select("timestamp").min().item()

        # no timestamps overlap in train and test
        assert test_min_ts > train_max_ts

