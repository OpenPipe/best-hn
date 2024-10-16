# To run this script: `echo train_reward_model.py | entr -s "uv run train_reward_model.py"`

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardTrainer, RewardConfig
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
from peft import TaskType

import wandb
from dotenv import load_dotenv

load_dotenv("/workspace/.env")

# Configuration
model_name = "unsloth/Meta-Llama-3.1-8B"
output_dir = "./models/train_reward_model_causal"
num_epochs = 1
batch_size = 1
learning_rate = 5e-5
max_length = 4096
eval_steps = 1000

wandb.init(project="reward_model_training")

print("Loading dataset...")
dataset: DatasetDict = load_dataset("OpenPipe/best-hn-comment-pairs")

TESTING = False
if TESTING:
    model_name = "unsloth/Llama-3.2-1B"
    output_dir = "./models/train_reward_model_causal_test"
    # Limit the training dataset to 1000 entries for testing
    dataset["train"] = dataset["train"].select(range(1000))
    dataset["validation"] = dataset["validation"].select(range(10))
    eval_steps = 10


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
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)

# model = AutoModelForSequenceClassification.from_pretrained(
#     model_name,
#     num_labels=1,
#     device_map="auto",
#     attn_implementation="flash_attention_2",
#     torch_dtype=torch.bfloat16,
# )


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

print("Configuring LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "lm_head"],
)
model = get_peft_model(model, peft_config)

# Configure training arguments
training_args = RewardConfig(
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    weight_decay=0,
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    logging_steps=50,
    save_strategy="steps",
    save_steps=1000,
    max_length=max_length,
    report_to="wandb",
    no_cuda=False,
    bf16=True,
    use_liger_kernel=True,
    warmup_steps=100,
    save_total_limit=1,
)

print("Initializing RewardTrainer...")
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    processing_class=tokenizer,
    compute_metrics=lambda x: {},  # TODO: implement
)

# print("Running initial evaluation on validation set...")
# eval_results = trainer.evaluate()
# print(f"Initial evaluation results: {eval_results}")

print("Starting model training...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Reward model training complete")
