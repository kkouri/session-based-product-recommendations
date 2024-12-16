import argparse
from pathlib import Path

import polars as pl
from beartype import beartype


@beartype
def get_max_ts(train_set_path: Path):
    return (
        pl.read_csv(train_set_path / 'train_set.csv', low_memory=True)
        .select("timestamp")
        .max()
        .item()
    )


@beartype
def main(train_set_path: Path, input_path: Path, output_path: Path, train_weeks: int, test_weeks: int):

    max_ts = get_max_ts(train_set_path)
    test_start = max_ts - test_weeks*7*24*60*60
    train_start = test_start - train_weeks*7*24*60*60

    print("Reading the dataset")
    # category tree
    df_schema = {"categoryid": pl.UInt32, "parentid": pl.UInt32}
    category_tree_df = (
        pl.read_csv(input_path / 'category_tree.csv', schema=df_schema, low_memory=True)
    )

    # item categories
    # we keep timestamp since there is possibility that the category changes over time
    df_schema = {"timestamp": pl.UInt64, "itemid": pl.UInt32, "property": pl.Utf8, "value": pl.Utf8}
    item_categories_df = (
        pl.concat([
            pl.read_csv(input_path / 'item_properties_part1.csv', schema=df_schema, low_memory=True),
            pl.read_csv(input_path / 'item_properties_part2.csv', schema=df_schema, low_memory=True)
        ])
        # Convert timestamp to seconds and cast to UInt32 to save memory
        .with_columns((pl.col("timestamp") // 1000).cast(pl.UInt32))
        # Filter all but categoryid
        .filter(pl.col("property") == "categoryid")
        .with_columns(categoryid=pl.col("value").cast(pl.UInt32))
        .drop(["property", "value"])
        # Find the parentid for each categoryid. If parentid is null, use categoryid as parentid
        .join(category_tree_df, on="categoryid", how="left")
        .with_columns(parentid=pl.col("parentid").fill_null(pl.col("categoryid")))
        # Filter by train timestamp
        .filter(pl.col("timestamp") >= train_start, pl.col("timestamp") < test_start)
        .drop("timestamp")
        .unique()
        .sort(["itemid", "categoryid", "parentid"])
    )

    print(item_categories_df)

    print("Saving the datasets")
    train_set_file = output_path / "item_categories.csv"
    item_categories_df.write_csv(train_set_file)

    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-set-path', type=Path, required=True)
    parser.add_argument('--input-path', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--train-weeks', type=int, default=3)
    parser.add_argument('--test-weeks', type=int, default=2)
    args = parser.parse_args()
    main(args.train_set_path, args.input_path, args.output_path, args.train_weeks, args.test_weeks)