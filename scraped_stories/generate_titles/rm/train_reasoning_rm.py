import logging
import os
import torch
import re
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
import wandb
import polars as pl
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from training_helpers import (
    compute_metrics,
    run_final_inference_and_report_metrics,
    MandT,
    create_dataset,
    serialize_story,
)
import s3fs
import typer
import numpy as np
from inference import ModelOrPath


def extract_reasoning_and_answer(text):
    """
    Extract reasoning and answer from text in the format:
    <think>...reasoning...</think><answer>number</answer>
    
    Returns:
        tuple: (reasoning, answer as float, has_reasoning, has_answer)
    """
    reasoning = None
    answer = None
    
    # Extract reasoning
    reasoning_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Extract answer
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        try:
            answer = float(answer_text)
        except ValueError:
            # Try to extract a number from text
            num_match = re.search(r"\d+", answer_text)
            if num_match:
                answer = float(num_match.group(0))
    
    return (
        reasoning,
        answer,
        reasoning is not None,
        answer is not None
    )


def create_reasoning_prompt(story):
    """Create a prompt that asks for reasoning before prediction."""
    base_story = serialize_story(story)
    
    prompt = f"""Given the following Hacker News submission, predict the number of upvotes it received. 
First reason through your prediction step by step, then provide your final numerical prediction.

Your response should follow this format exactly:
<think>
Your step-by-step reasoning here...
</think>
<answer>predicted_number_of_upvotes</answer>

Here is the submission:

{base_story}
"""
    return prompt


def create_reasoning_dataset(
    df: pl.DataFrame,
    split: str,
    num_rows: int,
    tokenizer,
    max_len: int,
    n_proc: int = 4,
):
    """Create a dataset with reasoning prompts."""
    df = df.filter(pl.col("split") == split).head(num_rows)
    
    # Add the log-transformed score as label
    df = df.with_columns(pl.col("score").log().alias("label"))
    
    # Create reasoning prompts
    df = df.with_columns(
        pl.struct(["title", "by", "time", "scraped_body", "url"])
        .map_elements(create_reasoning_prompt, return_dtype=pl.Utf8)
        .alias("text")
    )
    
    dataset = Dataset.from_polars(df.select(["text", "label"]))
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False)
    
    dataset = dataset.map(tokenize_function, batched=True, num_proc=n_proc)
    dataset = dataset.filter(lambda example: len(example["input_ids"]) <= max_len)
    return dataset


def compute_reasoning_metrics(eval_pred):
    """
    Compute metrics for the reasoning reward model.
    
    This extends the base metrics to also track reasoning quality metrics.
    """
    predictions, labels = eval_pred
    
    # Convert numpy arrays to torch tensors
    predictions = torch.tensor(predictions).squeeze()
    labels = torch.tensor(labels)
    
    # Filter out NaN values
    valid_indices = ~torch.isnan(predictions) & ~torch.isnan(labels)
    valid_predictions = predictions[valid_indices]
    valid_labels = labels[valid_indices]
    
    metrics = {
        "rmse": compute_metrics(eval_pred)["rmse"],
        "correlation": compute_metrics(eval_pred)["correlation"],
    }
    
    return metrics


def run_reasoning_inference_and_report_metrics(
    model_or_path: ModelOrPath, dataset: pl.DataFrame, output_dir: str = None
):
    """
    Run inference using the reasoning reward model and report metrics.
    
    This extends the base inference to also analyze reasoning quality.
    """
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
        from inference import load_model
        mandt = load_model(model_or_path)
        model = mandt.model
        
        if hasattr(model, "merge_and_unload"):
            print("Merging PEFT model with base model...")
            model = model.merge_and_unload()
        
        # Create reasoning prompts for inference
        dataset = dataset.with_columns(
            pl.struct(["title", "by", "time", "scraped_body", "url"])
            .map_elements(create_reasoning_prompt, return_dtype=pl.Utf8)
            .alias("text")
        )
        
        # Run inference
        predictions = run_inference_transformers(dataset["text"].to_list(), mandt)
        
        # Extract reasoning and answers
        reasoning_data = [extract_reasoning_and_answer(text) for text in predictions["text"]]
        reasoning_texts = [data[0] for data in reasoning_data]
        answer_values = [data[1] if data[1] is not None else float('nan') for data in reasoning_data]
        has_reasoning = [data[2] for data in reasoning_data]
        has_answer = [data[3] for data in reasoning_data]
        
        # Add predictions to dataset
        dataset = dataset.with_columns(
            pl.Series(name="predictions", values=answer_values),
            pl.Series(name="reasoning", values=reasoning_texts),
            pl.Series(name="has_reasoning", values=has_reasoning),
            pl.Series(name="has_answer", values=has_answer),
        )
        
        # Save predictions
        dataset.select("id", "predictions", "reasoning", "has_reasoning", "has_answer").write_parquet(predictions_path)
    
    # Calculate metrics
    from training_helpers import calculate_metrics_by_split
    metrics = calculate_metrics_by_split(dataset)
    
    # Calculate additional reasoning-specific metrics
    reasoning_presence = dataset.groupby("split").agg(
        pl.col("has_reasoning").mean().alias("reasoning_presence_rate"),
        pl.col("has_answer").mean().alias("answer_presence_rate"),
    )
    
    # Merge metrics
    metrics = metrics.join(reasoning_presence, on="split")
    
    print(metrics)
    
    # Log to wandb
    for row in metrics.iter_rows(named=True):
        split = row["split"]
        wandb.summary.update(
            {
                f"final/{split}/baseline_rmse": row["baseline_rmse"],
                f"final/{split}/model_rmse": row["model_rmse"],
                f"final/{split}/correlation": row["model_correlation"],
                f"final/{split}/reasoning_presence_rate": row["reasoning_presence_rate"],
                f"final/{split}/answer_presence_rate": row["answer_presence_rate"],
            }
        )
    
    return metrics


def run_inference_transformers(texts, mandt):
    """
    Run inference using the transformers model.
    
    This function captures both the prediction and the raw output text.
    """
    model = mandt.model
    tokenizer = mandt.tokenizer
    
    results = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
        
        with torch.no_grad():
            prediction = model(**inputs).logits.item()
        
        results.append({"prediction": prediction, "text": text})
    
    return pl.DataFrame(results)


def main(
    run_name: str = typer.Option(
        __file__.split("/")[-1].replace(".py", ""), help="Name of the run"
    ),
    num_epochs: int = typer.Option(1, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    gradient_accumulation_steps: int = typer.Option(
        4, help="Gradient accumulation steps"
    ),
    learning_rate: float = typer.Option(2e-4, help="Learning rate"),
    max_length: int = typer.Option(4096, help="Maximum token sequence length"),
    train_size: int = typer.Option(100, help="Number of training examples to use"),
    val_size: int = typer.Option(5, help="Number of validation examples to use"),
    base_model: str = typer.Option("Qwen/Qwen2.5-0.5B", help="Base model"),
):
    output_dir = f"./models/{run_name}"
    
    # Initialize wandb
    wandb.init(project="hn_scraped_stories", name=run_name)
    
    logging.info("Loading dataset...")
    df = pl.read_parquet(
        f"s3://{os.getenv('REMOTE_BUCKET')}/scraped-stories-filtered.parquet"
    )
    logging.info(f"Loaded {df.height} rows")
    
    logging.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        truncation=True,
        padding=True,
        max_length=max_length,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=1,  # Regression task
        device_map="auto",
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )
    _apply_liger_kernel_to_instance(model=model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": True}
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"
    
    logging.info("Configuring LoRA...")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="SEQ_CLS",
            r=8,
            lora_alpha=16,
            lora_dropout=0,
        ),
    )
    
    logging.info("Transforming datasets...")
    train_stories = create_reasoning_dataset(df, "train", train_size, tokenizer, max_length)
    print(f"Train stories: {len(train_stories)}")
    validation_stories = create_reasoning_dataset(df, "val", val_size, tokenizer, max_length)
    print(f"Validation stories: {len(validation_stories)}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0,
        evaluation_strategy="steps",
        eval_steps=0.05,
        logging_steps=100,
        save_strategy="steps",
        save_steps=1000,
        report_to="wandb",
        no_cuda=False,
        bf16=True,
        warmup_ratio=0.1,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    
    logging.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_stories,
        eval_dataset=validation_stories,
        processing_class=tokenizer,
        compute_metrics=compute_reasoning_metrics,
    )
    
    logging.info("Starting model training...")
    trainer.train()
    
    logging.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logging.info("Running final inference and reporting metrics...")
    metrics = run_reasoning_inference_and_report_metrics(
        MandT(model, tokenizer), df, output_dir
    )
    
    s3 = s3fs.S3FileSystem()
    
    logging.info(
        f"Uploading model to S3 at path s3://{os.getenv('REMOTE_BUCKET')}/models/{run_name}"
    )
    s3.put(
        output_dir,
        f"s3://{os.getenv('REMOTE_BUCKET')}/models/{run_name}",
        recursive=True,
        maxdepth=1,
    )
    
    logging.info("Model training complete")


if __name__ == "__main__":
    typer.run(main)