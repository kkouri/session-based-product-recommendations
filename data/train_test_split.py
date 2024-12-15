import polars as pl
import argparse
from pathlib import Path
from beartype import beartype


@beartype
def create_train_test_split(input_path: Path, output_path: Path, train_weeks: int, test_weeks: int):
    # define the schema of the dataframe
    df_schema = {"timestamp": pl.UInt64, "visitorid": pl.UInt32, "event": pl.Utf8, "itemid": pl.Utf8, "transactionid": pl.UInt32}

    events_df = (
        pl.read_csv(input_path, schema=df_schema, low_memory=True)
        # Convert timestamp to seconds and cast to UInt32 to save memory
        .with_columns((pl.col("timestamp")//1000).cast(pl.UInt32))
    )

    # Form sessions by splitting the events of a user to sessions based on time between the events
    # If the time between events is more than 30 minutes, it is considered a new session

    sessions_df = (
        events_df
        .sort(["visitorid", "timestamp"], descending=[False, False])
        .with_columns(
            previous_visitorid=pl.col("visitorid").shift(1),
            previous_timestamp=pl.col("timestamp").shift(1),
        )
        .with_columns(
            time_between_sec=pl.when(pl.col("previous_visitorid") == pl.col("visitorid")).then((pl.col("timestamp") - pl.col("previous_timestamp"))).otherwise(None)
        )
        .with_columns(
            is_session_boundary=(pl.col("time_between_sec").is_null() | (pl.col("time_between_sec") >= 30*60))
        )
        .with_columns(
            session = pl.col("is_session_boundary").cum_sum().cast(pl.UInt32),
        )
        .drop(["previous_visitorid", "previous_timestamp", "time_between_sec", "is_session_boundary"])
        # Count the amount of events in a session
        .with_columns(events_count=(pl.col("timestamp").rle_id().max().over("session") + 1))
        # Filter out sessions with only 1 event
        .filter(pl.col("events_count") > 1)
        .drop("events_count")
    )

    # Train test split

    max_ts = (
        sessions_df
        .select("timestamp")
        .max()
        .item()
    )

    # 2 weeks
    test_start = max_ts - test_weeks*7*24*60*60

    # 3 weeks
    train_start = test_start - train_weeks*7*24*60*60

    # We are only interested in sessions containing cart or order events
    sessions_with_cart_or_order = (
        sessions_df
        .filter(pl.col("event") != "view")
        .group_by("session")
        .agg(pl.col("event").unique().alias("unique_events"))
        .sort("session")
        .select("session")
    )

    # Train dataset is 3 weeks before the last week
    train_df = (
        sessions_df
        .filter((pl.col("timestamp") >= train_start) & (pl.col("timestamp") < test_start))
        # Count the amount of events in a session
        .with_columns(events_count=(pl.col("timestamp").rle_id().max().over("session") + 1))
        # Filter out sessions with only 1 event
        .filter(pl.col("events_count") > 1)
        .drop("events_count")
        # Filter out sessions with only clicks
        .join(sessions_with_cart_or_order, on="session", how="inner")
    )

    # Get unique train sessions
    train_sessions = train_df.select("session").unique()

    # Test dataset is 1 week before the last week
    test_df = (
        sessions_df
        .filter(pl.col("timestamp") >= test_start)
        # Filter out sessions that appear in the train dataset
        .join(train_sessions, on="session", how="anti")
        # Count the amount of events in a session
        .with_columns(events_count=(pl.col("timestamp").rle_id().max().over("session") + 1))
        # Filter out sessions with only 1 event
        .filter(pl.col("events_count") > 1)
        .drop("events_count")
        # Filter out sessions with only clicks
        .join(sessions_with_cart_or_order, on="session", how="inner")
    )

    # Save train dataset
    train_set_file = output_path / "train_set.csv"
    train_df.write_csv(train_set_file)

    # Save test dataset
    test_set_file = output_path / "test_set.csv"
    test_df.write_csv(test_set_file)


@beartype
def main(input_path: Path, output_path: Path, train_weeks: int, test_weeks: int):
    print("Creating train and test datasets")
    create_train_test_split(input_path, output_path, train_weeks, test_weeks)
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--train-weeks', type=int, default=3)
    parser.add_argument('--test-weeks', type=int, default=2)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.train_weeks, args.test_weeks)