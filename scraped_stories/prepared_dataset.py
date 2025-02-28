from datasets import load_dataset, DatasetDict, Dataset
from scraped_stories.cache import s3_cache
import polars as pl


def serialize_story(story):
    string = f"""<submitter>{story["by"]}</submitter>\n<url>{story["url"]}</url>\n<date>{story["time"].strftime("%Y-%m-%d")}</date>\n\n<body>{story["scraped_body"]}</body>\n<title>{story["title"]}</title>"""

    return string


@s3_cache.cache()
async def prepared_dataset():
    dataset: DatasetDict = load_dataset("OpenPipe/hacker-news-scraped-stories")  # type: ignore

    train_df = (
        dataset["train"]
        .to_polars()
        .with_columns(pl.Series(name="split", values=["train"] * len(dataset["train"])))  # type: ignore
    )
    val_df = (
        dataset["val"]
        .to_polars()
        .with_columns(pl.Series(name="split", values=["val"] * len(dataset["val"])))  # type: ignore
    )
    test_df = (
        dataset["test"]
        .to_polars()
        .with_columns(pl.Series(name="split", values=["test"] * len(dataset["test"])))  # type: ignore
    )

    df = pl.concat([train_df, val_df, test_df])

    df = df.filter((pl.col("scraping_error") == "no_error") & pl.col("text").is_null())

    df = df.filter(pl.col("time") > pl.datetime(2016, 1, 1))

    return df


def serialized(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.struct(["title", "by", "time", "scraped_body", "url"])
        .map_elements(serialize_story, return_dtype=pl.Utf8)
        .alias("serialized")
    )
