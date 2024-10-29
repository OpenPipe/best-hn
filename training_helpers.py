from typing import Optional
import torch
import polars as pl
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
import wandb
from inference import run_inference_transformers, ModelOrPath, MandT, load_model
from utils import stories_dataset, calculate_metrics_by_split


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
    model_or_path: ModelOrPath, output_dir: Optional[str] = None
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
        stories = stories_dataset().select(["id", "log_score", "split"])
        stories = stories.join(existing_predictions, on="id")
    except:
        # Original inference logic
        mandt = load_model(model_or_path)
        model = mandt.model
        tokenizer = mandt.tokenizer

        if hasattr(model, "merge_and_unload"):
            print("Merging PEFT model with base model...")
            model = model.merge_and_unload()

        stories = stories_dataset().select(["id", "log_score", "split", "serialized"])
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
