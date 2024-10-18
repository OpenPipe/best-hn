from typing import Union, Literal
import polars as pl
import dicttoxml
import tqdm
import requests
import os
import html
import re
import numpy as np


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

        def bust_cache():
            if path in cache:
                del cache[path]
            if os.path.exists(path):
                os.remove(path)
            print(f"Cache busted for {path}")

        wrapper.bust_cache = bust_cache
        return wrapper

    return decorator


@cache_dataframe("./data/full_dataset.parquet")
def full_dataset() -> pl.DataFrame:
    from datasets import load_dataset, Dataset

    dataset: Dataset = load_dataset("OpenPipe/hacker-news", split="train")

    return dataset.to_polars()


def unescape_html(text):
    unescaped = html.unescape(text).replace("<p>", "\n\n")
    return re.sub(r'<a href="([^"]+)"[^>]*>[^<]+</a>', r"\1", unescaped)


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

    progress_bar = tqdm.tqdm(total=len(comments_df), desc="Unescaping HTML")

    def unescape_html_wrapper(text):
        progress_bar.update(1)
        return unescape_html(text)

    comments_df = comments_df.with_columns(
        pl.col("text")
        .map_elements(unescape_html_wrapper, return_dtype=pl.Utf8)
        .alias("text")
    )

    return comments_df


def build_all_prompts(
    ids: Union[list[int], pl.Series], version: Literal["v1", "v2"]
) -> list[str]:
    if isinstance(ids, pl.Series):
        ids = ids.to_list()

    build_prompt = build_prompt_v1 if version == "v1" else build_prompt_v2

    prompts = []
    for id in tqdm.tqdm(ids, desc="Building prompts"):
        prompts.append(build_prompt(id))

    return prompts


def build_prompt_v1(comment_id: int) -> str:
    df = full_dataset()
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


def build_prompt_v2(comment_id: int) -> str:
    df = full_dataset()
    comment = df.row(comment_id, named=True)
    story = df.row(comment["top_level_parent"], named=True)

    data = {
        "story": {"title": story["title"]},
        "parent_chain": [],
        "comment": {
            "author": comment["by"],
            "text": comment["text"],
        },
    }

    current_parent = df.row(comment["parent"], named=True)
    while current_parent["id"] != story["id"]:
        data["parent_chain"].append(
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


@cache_dataframe("./data/stories_dataset.parquet")
def stories_dataset() -> pl.DataFrame:
    stories = full_dataset().filter(
        (pl.col("type") == "story")
        & pl.col("time").is_not_null()
        & pl.col("text").is_not_null()
        & pl.col("url").is_null()
        & pl.col("deleted").is_null()
        & pl.col("dead").is_null()
    )

    # There's a weird discontinuity in late 2015, just ignore it
    stories = stories.filter(pl.col("time") >= pl.datetime(2016, 1, 1))

    # Add a log score, it's a very skewed distribution
    stories = stories.with_columns(pl.col("score").log().alias("log_score"))

    progress_bar = tqdm.tqdm(total=len(stories), desc="Serializing stories")

    def serialize_story(story):
        progress_bar.update(1)
        return f"""
{story["title"]}
{story["by"]}, {story["time"].strftime("%Y-%m-%d")}

{html.unescape(story["text"]).replace("<p>", "\n\n")}
"""

    stories = stories.with_columns(
        pl.struct(["title", "by", "time", "text"])
        .map_elements(serialize_story, return_dtype=pl.Utf8)
        .alias("serialized")
    )

    progress_bar.close()

    stories = stories.sample(fraction=1, shuffle=True, seed=42)

    split_assignments = np.random.choice(
        ["train", "test", "val"], size=len(stories), p=[0.8, 0.1, 0.1]
    )

    stories = stories.with_columns(pl.Series("split", split_assignments))

    return stories.select(
        "id",
        "title",
        "by",
        "text",
        "score",
        "descendants",
        "time",
        "log_score",
        "serialized",
        "split",
    )
