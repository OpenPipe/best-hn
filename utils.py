import polars as pl
from functools import lru_cache
import dicttoxml


@lru_cache(maxsize=None)
def dataset() -> pl.DataFrame:
    from datasets import load_dataset, Dataset

    dataset: Dataset = load_dataset("OpenPipe/hacker-news", split="train")

    return dataset.to_polars()


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
