from typing import Optional
import torch
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
import wandb
from .inference import run_inference_transformers, ModelOrPath, MandT, load_model
import math
import logging
from datasets import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
    model_or_path: ModelOrPath, dataset=None, output_dir: Optional[str] = None
):
    if output_dir is None:
        if not isinstance(model_or_path, str):
            raise ValueError(
                "output_dir is required when model_or_path is not a path string. Please provide an output directory."
            )
        output_dir = model_or_path

    predictions_path = f"{output_dir}/dataset_predictions.parquet"

    if dataset is None:
        raise ValueError("dataset is required")

    # Check if predictions file already exists
    try:
        existing_predictions = pl.read_parquet(predictions_path)
        print(f"Loading existing predictions from {predictions_path}")
        stories = dataset.select(["id", "log_score", "split"])
        stories = stories.join(existing_predictions, on="id")
    except:
        # Original inference logic
        mandt = load_model(model_or_path)
        model = mandt.model
        tokenizer = mandt.tokenizer

        if hasattr(model, "merge_and_unload"):
            print("Merging PEFT model with base model...")
            model = model.merge_and_unload()

        stories = dataset.select(["id", "log_score", "split", "serialized"])
        predictions = run_inference_transformers(stories["serialized"].to_list(), mandt)
        stories = stories.with_columns(
            pl.Series(name="predictions", values=predictions)
        )
        stories.select("id", "predictions").write_parquet(predictions_path)

    metrics = calculate_metrics_by_split(stories)

    print(metrics)

    # Log metrics to wandb if it's being used
    if wandb.run is not None:
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


def create_dataset(
    df: pl.DataFrame,
    split: str,
    num_rows: int,
    tokenizer,
    max_len: int,
    n_proc: int = 4,
):
    df = df.with_columns(pl.col("score").log().alias("log_score"))
    df = df.filter(pl.col("split") == split).head(num_rows)
    df = df.with_columns(
        [
            pl.col("serialized").alias("text"),
            pl.col("log_score").alias("label"),
        ]
    )
    dataset = Dataset.from_polars(df.select(["text", "label"]))

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)

    dataset = dataset.map(tokenize_function, batched=True, num_proc=n_proc)
    dataset = dataset.filter(lambda example: len(example["input_ids"]) <= max_len)
    return dataset
