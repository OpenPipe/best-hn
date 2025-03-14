import os
from panza import SQLiteCache
from dotenv import load_dotenv
from datasets import load_dataset, Dataset

load_dotenv()

cache_db_path = os.path.join(os.path.dirname(__file__), "shared_cache.db")
cache = SQLiteCache(cache_db_path)


def pull_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
) -> Dataset:
    print(f"Loading dataset from HuggingFace (max {max_items} items)...")
    dataset: Dataset = load_dataset(
        "OpenPipe/hacker-news-scraped-stories-filtered", split=split
    )  # type: ignore
    dataset = dataset.filter(lambda x: x["score"] >= min_score)
    dataset = dataset.select(range(max_items))

    return dataset


def prompt_for_title(content: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that generates engaging titles for Hacker News posts. Respond with just the title, no other text.",
        },
        {
            "role": "user",
            "content": f"Generate a concise, engaging title for this Hacker News submission. The title should be informative yet catchy.\n\nContent: {content}",
        },
    ]
