# To run this script: `echo train_reward_model.py | entr -s "uv run train_reward_model.py"`


from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardTrainer, RewardConfig
from peft.tuners.lora import LoraConfig
from peft.mapping import get_peft_model
import wandb

# Configuration
model_name = "unsloth/Llama-3.2-3B"
dataset_path = "./data/sample_pairs"
output_dir = "./reward_model_output"
num_epochs = 1
batch_size = 1  # For some reason making this larger doesn't help training time, why?
learning_rate = 5e-5
max_length = 4096

# Initialize wandb
wandb.init(project="reward_model_training")

print("Loading dataset...")
dataset: Dataset = load_from_disk(dataset_path)["train"]

# Limit the dataset to 10,000 examples
dataset = dataset.select(range(10000))


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
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    device_map="auto",
)

print(f"Tokenizer padding token: {tokenizer.pad_token}")
print(f"Model padding token: {model.config.pad_token_id}")

model.config.pad_token_id = tokenizer.pad_token_id
tokenizer.padding_side = "right"

print("Processing dataset...")
processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
)

print("Splitting dataset...")
train_val_split = processed_dataset.train_test_split(test_size=500, seed=42)

print("Configuring LoRA...")
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=16,
    lora_dropout=0,
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
    eval_steps=500,
    logging_steps=100,
    save_strategy="steps",
    save_steps=1000,
    # load_best_model_at_end=True,
    max_length=max_length,
    report_to="wandb",
    no_cuda=False,
    bf16=True,
)

print("Initializing RewardTrainer...")
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_val_split["train"],
    eval_dataset=train_val_split["test"],
    tokenizer=tokenizer,
)

print("Starting model training...")
trainer.train()

print("Saving final model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Reward model training complete")
