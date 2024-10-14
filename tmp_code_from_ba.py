import json

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardConfig, RewardTrainer

import wandb

from dotenv import load_dotenv

load_dotenv("/workspace/.env")


print(f"Total GPUs: {torch.cuda.device_count()}")

wandb.init(project="ensure_same_perf")

tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Llama-3.2-3B",
    trust_remote_code=True,
    truncation=True,
    padding=True,
    max_length=8192,
)

split_dataset: DatasetDict = load_dataset("OpenPipe/test-reward-dataset-tmp-delete-me")

model = AutoModelForSequenceClassification.from_pretrained(
    "unsloth/Llama-3.2-3B",
    num_labels=1,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model.config.pad_token_id = tokenizer.pad_token_id

peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0,
)
model = get_peft_model(model, peft_config)

trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=RewardConfig(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=0.00005,
        logging_steps=5,
        evaluation_strategy="steps",
        eval_steps=0.1,
        optim="adamw_bnb_8bit",
        weight_decay=0,
        lr_scheduler_type="linear",
        output_dir="/tmp/ignore_dir",
        report_to="wandb",
        remove_unused_columns=False,
        max_length=8192,
        bf16=True,
    ),
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
)

print(f"Trainer args for model:")
print(json.dumps(trainer.args.to_dict(), indent=2))

trainer.train()
# Save the model
output_dir = "/tmp/reward_model"

print("Merging and unloading model")
model = model.merge_and_unload()

print("Saving model")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Reward model training complete")
