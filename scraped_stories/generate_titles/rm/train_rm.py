import logging
import os
import torch
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
)
import s3fs
import typer


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
    train_stories = create_dataset(df, "train", train_size, tokenizer, max_length)
    print(f"Train stories: {len(train_stories)}")
    validation_stories = create_dataset(df, "val", val_size, tokenizer, max_length)
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
        compute_metrics=compute_metrics,
    )

    logging.info("Starting model training...")
    trainer.train()

    logging.info("Saving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logging.info("Running final inference and reporting metrics...")
    metrics = run_final_inference_and_report_metrics(
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
