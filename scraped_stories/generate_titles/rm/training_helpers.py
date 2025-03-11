from typing import Optional
import torch  # type: ignore
import polars as pl  # type: ignore
from scipy.stats import pearsonr  # type: ignore
from sklearn.metrics import root_mean_squared_error  # type: ignore
import wandb  # type: ignore
from inference import run_inference_transformers, ModelOrPath, MandT, load_model
import math
import logging
from datasets import Dataset  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def serialize_story(story):
    string = f"""<submitter>{story["by"]}</submitter>\n<url>{story["url"]}</url>\n<date>{story["time"].strftime("%Y-%m-%d")}</date>\n\n<body>{story["scraped_body"]}</body>\n<title>{story["title"]}</title>"""

    return string


def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Convert numpy arrays to torch tensors
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels)

    # Filter out NaN values
    valid_indices = ~torch.isnan(predictions) & ~torch.isnan(labels)
    valid_predictions = predictions[valid_indices]
    valid_labels = labels[valid_indices]

    return {
        "rmse": root_mean_squared_error(valid_labels, valid_predictions),
        "correlation": pearsonr(valid_labels, valid_predictions)[0],
    }


def run_final_inference_and_report_metrics(
    model_or_path: ModelOrPath, dataset: pl.DataFrame, output_dir: Optional[str] = None
):
    if output_dir is None:
        if not isinstance(model_or_path, str):
            raise ValueError(
                "output_dir is required when model_or_path is not a path string. Please provide an output directory."
            )
        output_dir = model_or_path

    predictions_path = f"{output_dir}/dataset_predictions.parquet"

    # Check if predictions file already exists
    try:
        existing_predictions = pl.read_parquet(predictions_path)
        print(f"Loading existing predictions from {predictions_path}")
        stories = dataset.select(["id", "label", "split"])
        stories = stories.join(existing_predictions, on="id")
    except:
        mandt = load_model(model_or_path)
        model = mandt.model

        if hasattr(model, "merge_and_unload"):
            print("Merging PEFT model with base model...")
            model = model.merge_and_unload()

        dataset = with_training_columns(dataset)

        predictions = run_inference_transformers(dataset["text"].to_list(), mandt)
        dataset = dataset.with_columns(
            pl.Series(name="predictions", values=predictions)
        )
        dataset.select("id", "predictions").write_parquet(predictions_path)

    metrics = calculate_metrics_by_split(dataset)

    print(metrics)

    for row in metrics.iter_rows(named=True):
        split = row["split"]
        wandb.summary.update(
            {
                f"final/{split}/baseline_rmse": row["baseline_rmse"],
                f"final/{split}/model_rmse": row["model_rmse"],
                f"final/{split}/correlation": row["model_correlation"],
            }
        )

    return metrics


def calculate_metrics_by_split(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate correlation and RMSE metrics for each split in the dataset.

    Args:
        df: DataFrame with label, predictions and split columns

    Returns:
        DataFrame with metrics for each split
    """
    metrics = []

    for split in df["split"].unique():
        split_df = df.filter(pl.col("split") == split)

        # Calculate baseline (mean) metrics
        average_score = split_df["label"].mean()
        rmse_baseline = math.sqrt(
            (split_df["label"] - average_score).pow(2).sum() / len(split_df)
        )

        # Calculate model metrics
        rmse_model = math.sqrt(
            (split_df["label"] - split_df["predictions"]).pow(2).sum() / len(split_df)
        )
        correlation_model = split_df.select(pl.corr("label", "predictions"))["label"][0]

        metrics.append(
            {
                "split": split,
                "baseline_rmse": rmse_baseline,
                "model_rmse": rmse_model,
                "model_correlation": correlation_model,
                "num_rows": len(split_df),
            }
        )

    return pl.DataFrame(metrics)


def with_training_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        [
            pl.col("score").log().alias("label"),
            pl.struct(["title", "by", "time", "scraped_body", "url"])
            .map_elements(serialize_story, return_dtype=pl.Utf8)
            .alias("text"),
        ]
    )


def create_dataset(
    df: pl.DataFrame,
    split: str,
    num_rows: int,
    tokenizer,
    max_len: int,
    n_proc: int = 4,
):
    df = df.filter(pl.col("split") == split).head(num_rows)
    df = with_training_columns(df)
    dataset = Dataset.from_polars(df.select(["text", "label"]))

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    dataset = dataset.map(tokenize_function, batched=True, num_proc=n_proc)
    dataset = dataset.filter(lambda example: len(example["input_ids"]) <= max_len)
    return dataset
