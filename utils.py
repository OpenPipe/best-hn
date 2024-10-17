from typing import Union
import polars as pl
from functools import lru_cache
import dicttoxml
import tqdm
import requests
import os


def cache_dataframe(path):
    def decorator(func):
        cache = {}

        def wrapper(*args, **kwargs):
            if path in cache:
                return cache[path]
            if os.path.exists(path):
                df = pl.read_parquet(path)
            else:
                df = func(*args, **kwargs)
                print(f"Caching dataframe to {path}")
                df.write_parquet(path)
            cache[path] = df
            return df

        return wrapper

    return decorator


@cache_dataframe("./data/full_dataset.parquet")
def full_dataset() -> pl.DataFrame:
    from datasets import load_dataset, Dataset

    dataset: Dataset = load_dataset("OpenPipe/hacker-news", split="train")

    return dataset.to_polars()


@cache_dataframe("./data/augmented_comments.parquet")
def augmented_comments() -> pl.DataFrame:
    df = full_dataset()

    comments_df = df.filter((pl.col("type") == "comment"))

    comments_df = comments_df.select(
        pl.col("id", "by", "time", "text", "parent", "top_level_parent", "kids")
    )

    # add a new column siblings_count
    comments_df = comments_df.with_columns(
        [pl.col("id").count().over("parent").alias("siblings_count")]
    )

    sibling_ranks = dict()

    # iterate over the rows of the dataframe
    for groupable_comment in df.select(pl.col("kids")).iter_rows():
        kids = groupable_comment[0]
        if kids is not None:
            for i, kid in enumerate(kids):
                sibling_ranks[kid] = i + 1

    # Find the maximum value in the sibling_ranks dictionary
    max_sibling_rank = max(sibling_ranks.values())

    # Print the maximum value
    print(f"The maximum sibling rank is: {max_sibling_rank}")
    print(f"The number of sibling ranks is: {len(sibling_ranks)}")

    comments_df = comments_df.with_columns(
        [
            pl.col("id")
            .replace_strict(
                list(sibling_ranks.keys()), list(sibling_ranks.values()), default=-1
            )
            .alias("sibling_rank")
        ]
    )
    del sibling_ranks

    comments_df = comments_df.with_columns(pl.lit(-1).alias("nested_level"))

    comments_df = comments_df.with_columns(
        pl.when(pl.col("top_level_parent") == pl.col("parent"))
        .then(0)
        .otherwise(pl.col("nested_level"))
        .alias("nested_level")
    )

    for i in tqdm.tqdm(range(1, 25), desc="Calculating nested levels"):
        parent_ids = comments_df.filter(pl.col("nested_level") == i - 1)["id"]

        comments_df = comments_df.with_columns(
            pl.when(pl.col("parent").is_in(parent_ids))
            .then(i)
            .otherwise(pl.col("nested_level"))
            .alias("nested_level")
        )

    return comments_df


def build_all_prompts(ids: Union[list[int], pl.Series]) -> list[str]:
    if isinstance(ids, pl.Series):
        ids = ids.to_list()

    prompts = []
    for id in tqdm.tqdm(ids, desc="Building prompts"):
        prompts.append(build_prompt(id))

    return prompts


def build_prompt(comment_id: int) -> str:
    df = dataset()
    comment = df.row(comment_id, named=True)
    story = df.row(comment["top_level_parent"], named=True)

    data = {
        "instructions": "Your goal is to analyze the following comment and estimate how highly it will be upvoted by the Hacker News community.",
        "comment": {
            "author": comment["by"],
            "text": comment["text"],
            "parent_chain": [],
        },
        "story": {"title": story["title"]},
    }

    current_parent = df.row(comment["parent"], named=True)
    while current_parent["id"] != story["id"]:
        data["comment"]["parent_chain"].append(
            {"author": current_parent["by"], "text": current_parent["text"]}
        )
        current_parent = df.row(current_parent["parent"], named=True)

    if story["url"] is not None:
        data["story"]["url"] = story["url"]
    if story["text"] is not None:
        data["story"]["text"] = story["text"]

    xml: bytes = dicttoxml.dicttoxml(data, attr_type=False, root=False)

    return xml.decode("utf-8")


def run_inference_sglang(
    prompts: Union[list[str], pl.Series], chunk_size: int = 100
) -> list[float]:
    if isinstance(prompts, pl.Series):
        prompts = prompts.to_list()

    # Chunk prompts into lists of INFERENCE_CHUNK_SIZE
    chunks = [prompts[i : i + chunk_size] for i in range(0, len(prompts), chunk_size)]

    rewards = []
    for chunk in tqdm.tqdm(chunks, desc="Running inference"):
        json_data = {
            "conv": chunk,
        }
        response = requests.post("http://127.0.0.1:30000/judge", json=json_data).json()
        rewards.extend([x["embedding"][0] for x in response])

    return rewards


def with_story_info(comments_df: pl.DataFrame) -> pl.DataFrame:
    stories_df = (
        dataset()
        .filter(pl.col("type") == "story")
        .select(pl.col("id", "title", "url"))
        .rename(
            {
                "id": "story_id",
                "title": "story_title",
                "url": "story_url",
            }
        )
    )

    return comments_df.join(
        stories_df, left_on="top_level_parent", right_on="story_id", how="left"
    )
