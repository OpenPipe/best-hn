# uv run modal run --detach scraped_stories.rm.model1

from .modal_app import app
import modal
import logging

s3_bucket_name = "[placeholder]"

if modal.is_local():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    s3_bucket_name = os.getenv("REMOTE_BUCKET", "[placeholder]")
    logging.info(f"Using S3 bucket: {s3_bucket_name}")


@app.function(
    secrets=[modal.Secret.from_dotenv(".env")],
    gpu="H100",
    memory=32768 * 2,
    timeout=3600 * 24,
    volumes={
        "/remote": modal.CloudBucketMount(
            bucket_name=s3_bucket_name,
            secret=modal.Secret.from_dotenv(".env"),
            read_only=False,
        )
    },
)
async def main():
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
    from .training_helpers import (
        compute_metrics,
        run_final_inference_and_report_metrics,
        MandT,
        create_dataset,
    )

    # Configuration
    base_model = "unsloth/Meta-Llama-3.1-8B"
    run_name = __file__.split("/")[-1].replace(".py", "")
    output_dir = f"/remote/rm/models/{run_name}"
    num_epochs = 1
    batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_length = 4096

    # Initialize wandb
    wandb.init(project="hn_scraped_stories", name=run_name)

    logging.info("Loading dataset...")
    df = pl.read_parquet(
        f"s3://{os.getenv('REMOTE_BUCKET')}/scraped-stories-with-datetime.parquet"
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
    tokenizer.padding_side = "right"

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
    train_stories = create_dataset(df, "train", 1000000, tokenizer, max_length)
    print(f"Train stories: {len(train_stories)}")
    validation_stories = create_dataset(df, "val", 500, tokenizer, max_length)
    print(f"Validation stories: {len(validation_stories)}")

    # Configure training arguments
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
        tokenizer=tokenizer,
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

    logging.info("Model training complete")


@app.local_entrypoint()
def main_local():
    print("Running main locally")
    main.remote()
