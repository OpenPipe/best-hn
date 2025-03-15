import polars as pl
import math
from pydantic import BaseModel, Field, field_serializer
from datetime import datetime
from typing import Optional, Dict
import os
from panza import limit_concurrency
import httpx
from ..utils import cache


class ScoreRequest(BaseModel):
    title: str = Field(..., description="The title of the story")
    by: str = Field(..., description="The submitter of the story")
    time: str = Field(..., description="The submission time of the story")
    scraped_body: str = Field(..., description="The body content of the story")
    url: Optional[str] = Field(None, description="The URL of the story")

    @field_serializer("time")
    def serialize_time(self, value: datetime) -> str:
        if isinstance(value, str):
            return value
        return value.isoformat()


def serialize_story(story):
    string = f"""<submitter>{story["by"]}</submitter>\n<url>{story["url"]}</url>\n<date>{story["time"].strftime("%Y-%m-%d")}</date>\n\n<body>{story["scraped_body"]}</body>\n<title>{story["title"]}</title>"""

    return string


def with_serialized_stories(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.struct(["title", "by", "time", "scraped_body", "url"])
        .map_elements(serialize_story, return_dtype=pl.Utf8)
        .alias("serialized")
    )


def calculate_metrics_by_split(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate correlation and RMSE metrics for each split in the dataset.

    Args:
        df: DataFrame with log_score, predictions and split columns

    Returns:
        DataFrame with metrics for each split
    """
    metrics = []

    for split in df["split"].unique():
        split_df = df.filter(pl.col("split") == split)

        # Calculate baseline (mean) metrics
        average_score = split_df["log_score"].mean()
        rmse_baseline = math.sqrt(
            (split_df["log_score"] - average_score).pow(2).sum() / len(split_df)
        )

        # Calculate model metrics
        rmse_model = math.sqrt(
            (split_df["log_score"] - split_df["predictions"]).pow(2).sum()
            / len(split_df)
        )
        correlation_model = split_df.select(pl.corr("log_score", "predictions"))[
            "log_score"
        ][0]

        metrics.append(
            {
                "split": split,
                "baseline_rmse": rmse_baseline,
                "model_rmse": rmse_model,
                "model_correlation": correlation_model,
            }
        )

    return pl.DataFrame(metrics)


REWARD_MODEL_URL = os.getenv("REWARD_MODEL_URL", "http://localhost:80/score")

print(f"REWARD_MODEL_URL: {REWARD_MODEL_URL}")


@cache.cache()
async def score_title(story_dict: Dict, _reward_model: str) -> float:
    """Get the reward model score for a story asynchronously.

    Args:
        story_dict: Dictionary containing story data with keys: title, by, time, scraped_body, url
        _reward_model: Identifier for the reward model (unused here).

    Returns:
        The score returned by the reward model.
    """
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            # Clone the story_dict to avoid modifying the original
            request_dict = story_dict.copy()
            request_dict["time"] = request_dict["time"].isoformat()
            response = await client.post(
                REWARD_MODEL_URL, json=ScoreRequest(**request_dict).model_dump()
            )
            response.raise_for_status()
            data = response.json()
            return data["score"]
        except httpx.TimeoutException:
            print(f"Timeout connecting to reward model at {REWARD_MODEL_URL}")
            return 0.0  # Return a default score on timeout
        except Exception as e:
            print(f"Error connecting to reward model: {str(e)}")
            return 0.0  # Return a default score on error
