import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
import wandb
from dotenv import load_dotenv
import polars as pl
from utils import stories_dataset
from liger_kernel.transformers import _apply_liger_kernel_to_instance
from training_helpers import compute_metrics
import os

load_dotenv("/workspace/.env")

# Configuration
base_model = "unsloth/Meta-Llama-3.1-8B"
run_name = "stories_train_model_v6"
output_dir = f"./models/{run_name}"
num_epochs = 1
batch_size = 4
gradient_accumulation_steps = 4
learning_rate = 2e-4
max_length = 4096

# Initialize wandb
wandb.init(project="hn_stories_model_training", name=run_name)


def create_dataset(split, num_rows, tokenizer):
    stories = stories_dataset()
    stories = stories.filter(pl.col("split") == split).head(num_rows)

    stories = stories.with_columns(
        [
            pl.col("serialized").alias("text"),
            pl.col("log_score").alias("label"),
        ]
    )

    stories = stories.with_columns(
        [
            pl.col("text")
            .map_elements(
                lambda x: tokenizer(x)["input_ids"], return_dtype=pl.List(pl.Int64)
            )
            .alias("input_ids"),
        ]
    ).select(["input_ids", "label"])
    return Dataset.from_polars(stories)


print("Loading tokenizer and model...")
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

# Add this block to freeze all parameters except the classification head
for param in model.parameters():
    param.requires_grad = False
for param in model.score.parameters():
    param.requires_grad = True

model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

print("Loading dataset...")
train_stories = create_dataset("train", 1000000, tokenizer)
validation_stories = create_dataset("val", 1000, tokenizer)


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
    warmup_steps=100,
    gradient_accumulation_steps=gradient_accumulation_steps,
)


class ClassificationHeadTrainer(Trainer):
    def _save(self, output_dir: str, state_dict=None):
        # Only save the classification head parameters
        if state_dict is None:
            state_dict = self.model.state_dict()

        head_state_dict = {
            k: v for k, v in state_dict.items() if k.startswith("score.")
        }

        os.makedirs(output_dir, exist_ok=True)
        torch.save(head_state_dict, os.path.join(output_dir, "classification_head.bin"))

    def _load_state_dict_in_model(self, state_dict):
        # Load only classification head parameters
        self.model.score.load_state_dict(state_dict)


print("Initializing Trainer...")
trainer = ClassificationHeadTrainer(
    model=model,
    args=training_args,
    train_dataset=train_stories,
    eval_dataset=validation_stories,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting model training...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Stories model training complete")
