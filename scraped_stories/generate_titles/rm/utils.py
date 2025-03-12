import polars as pl
import math
from pydantic import BaseModel, Field, field_serializer
from datetime import datetime
from typing import Optional


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
