import unsloth
from vllm import SamplingParams
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import re
from datasets import load_dataset, Dataset
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv
import typer
import wandb

load_dotenv()

max_seq_length = 8192
lora_rank = 16  # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-14B-Instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,  # False for LoRA 16bit
    fast_inference=True,  # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.7,  # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)


# Load and prep dataset
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]  # type: ignore
    data = data.map(
        lambda x: {  # type: ignore
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )  # type: ignore
    return data  # type: ignore


dataset = get_gsm8k_questions()


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def main(
    run_name: str = typer.Option(
        __file__.split("/")[-1].replace(".py", ""), help="Run name"
    ),
    max_seq_length: int = typer.Option(8192, help="Maximum sequence length"),
    lora_rank: int = typer.Option(16, help="LoRA rank"),
    base_model: str = typer.Option(
        "unsloth/Qwen2.5-14B-Instruct", help="Base model to use"
    ),
    load_in_4bit: bool = typer.Option(True, help="Load model in 4bit mode"),
    fast_inference: bool = typer.Option(True, help="Enable fast inference"),
    gpu_memory_utilization: float = typer.Option(
        0.7, help="GPU memory utilization ratio"
    ),
    learning_rate: float = typer.Option(5e-6, help="Learning rate"),
    adam_beta1: float = typer.Option(0.9, help="Adam beta1"),
    adam_beta2: float = typer.Option(0.99, help="Adam beta2"),
    weight_decay: float = typer.Option(0.1, help="Weight decay"),
    warmup_ratio: float = typer.Option(0.1, help="Warmup ratio"),
    lr_scheduler_type: str = typer.Option(
        "cosine", help="Learning rate scheduler type"
    ),
    optim: str = typer.Option("paged_adamw_8bit", help="Optimizer"),
    logging_steps: int = typer.Option(1, help="Logging steps"),
    per_device_train_batch_size: int = typer.Option(
        1, help="Per-device training batch size"
    ),
    gradient_accumulation_steps: int = typer.Option(
        1, help="Gradient accumulation steps"
    ),
    num_generations: int = typer.Option(6, help="Number of generations"),
    max_prompt_length: int = typer.Option(256, help="Max prompt length"),
    max_completion_length: int = typer.Option(200, help="Max completion length"),
    max_steps: int = typer.Option(100, help="Maximum training steps"),
    save_steps: int = typer.Option(250, help="Steps interval for saving"),
    max_grad_norm: float = typer.Option(0.1, help="Maximum gradient norm"),
    output_dir: str = typer.Option("outputs", help="Output directory"),
) -> None:
    wandb.init(project="hn_title_generation", name=run_name)
    # Load model and tokenizer using the provided hyperparameters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        fast_inference=fast_inference,
        max_lora_rank=lora_rank,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Load dataset
    dataset = get_gsm8k_questions()

    # Set up GRPO training arguments using command-line hyperparameters
    training_args = GRPOConfig(
        use_vllm=True,
        learning_rate=learning_rate,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim=optim,
        logging_steps=logging_steps,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        save_steps=save_steps,
        max_grad_norm=max_grad_norm,
        report_to="wandb",
        output_dir=output_dir,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    # Test generation
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "Which is bigger? 9.11 or 9.9?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )
    output_text = (
        model.fast_generate(
            [text],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )
    print("Output:", output_text)

    model.save_lora("grpo_saved_lora")

    text2 = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Which is bigger? 9.11 or 9.9?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    output_text2 = (
        model.fast_generate(
            text2,
            sampling_params=sampling_params,
            lora_request=model.load_lora("grpo_saved_lora"),
        )[0]
        .outputs[0]
        .text
    )
    print("Output after loading LoRA:", output_text2)


if __name__ == "__main__":
    typer.run(main)
