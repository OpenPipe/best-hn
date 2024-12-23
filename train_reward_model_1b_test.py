# To run this script: `echo train_reward_model.py | entr -s "uv run train_reward_model.py"`

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardTrainer, RewardConfig
import wandb
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

# Configuration
model_name = "unsloth/Llama-3.2-1B"
output_dir = "./models/reward_model_1b_test"
num_epochs = 1
batch_size = 1  # For some reason making this larger doesn't help training time, why?
learning_rate = 5e-5
max_length = 4096

# Initialize wandb
wandb.init(project="reward_model_training")

print("Loading dataset...")
dataset: DatasetDict = load_dataset("OpenPipe/best-hn-comment-pairs")

dataset["train"] = dataset["train"].shuffle(seed=42).select(range(100))
dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(100))


def preprocess_function(examples):
    chosen = examples["chosen_prompt"]
    rejected = examples["rejected_prompt"]

    chosen_tokens = tokenizer(
        chosen, truncation=True, padding="max_length", max_length=max_length
    )
    rejected_tokens = tokenizer(
        rejected, truncation=True, padding="max_length", max_length=max_length
    )

    return {
        "input_ids_chosen": chosen_tokens["input_ids"],
        "attention_mask_chosen": chosen_tokens["attention_mask"],
        "input_ids_rejected": rejected_tokens["input_ids"],
        "attention_mask_rejected": rejected_tokens["attention_mask"],
    }


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.chat_template = "{{ messages[0]['content'] }}"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

print(f"Tokenizer padding token: {tokenizer.pad_token}")
print(f"Model padding token: {model.config.pad_token_id}")

model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

print("Processing dataset...")
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# Configure training arguments
training_args = RewardConfig(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_strategy="steps",
    save_steps=2000,
    # load_best_model_at_end=True,
    max_length=max_length,
    report_to="wandb",
    no_cuda=False,
    bf16=True,
    use_liger_kernel=True,
    warmup_steps=100,
)

print("Initializing RewardTrainer...")
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    tokenizer=tokenizer,
)

print("Starting model training...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Model saved to {output_dir}")
