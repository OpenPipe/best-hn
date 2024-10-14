# To run this script: `echo train_reward_model.py | entr -s "uv run train_reward_model.py"`

import torch
import json
from datasets import load_from_disk, DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardTrainer, RewardConfig
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
import wandb
import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

# Add logging to check HF_TOKEN
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("HF_TOKEN is set in the environment.")
    api = HfApi()
    try:
        user_info = api.whoami(token=hf_token)
        print(f"Logged in as: {user_info['name']}")
    except Exception as e:
        print(f"Error verifying HuggingFace login: {e}")
else:
    print("HF_TOKEN is not set in the environment.")
    print("All variables: ", os.environ)

# Configuration
model_name = "unsloth/Llama-3.2-3B"
output_dir = "./reward_model_output"
num_epochs = 2
batch_size = 1  # For some reason making this larger doesn't help training time, why?
learning_rate = 5e-5
max_length = 8192
gradient_accumulation_steps = 4

# Initialize wandb
wandb.init(project="ensure_same_perf")

print("Loading dataset...")
dataset: DatasetDict = load_dataset("OpenPipe/test-reward-dataset-tmp-delete-me")


print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, padding=True, truncation=True, max_length=max_length
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Tokenizer padding token: {tokenizer.pad_token}")
print(f"Model padding token: {model.config.pad_token_id}")

model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

print("Configuring LoRA...")
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0,
)
model = get_peft_model(model, peft_config)

print("Model config: ", model.config)

# Configure training arguments
training_args = RewardConfig(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0,
    evaluation_strategy="steps",
    eval_steps=0.1,
    logging_steps=5,
    save_strategy="steps",
    save_steps=500,
    # load_best_model_at_end=True,
    max_length=max_length,
    report_to="wandb",
    no_cuda=False,
    bf16=True,
    use_liger_kernel=False,
    warmup_steps=10,
    optim="adamw_bnb_8bit",
    gradient_accumulation_steps=gradient_accumulation_steps,
)

print("Initializing RewardTrainer...")
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],  # Use "test" split for validation
    tokenizer=tokenizer,
)

print(f"Trainer args for model:")
print(json.dumps(trainer.args.to_dict(), indent=2))


print("Starting model training...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Reward model training complete")
