import unsloth
from vllm import SamplingParams
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import RewardFunc
from dotenv import load_dotenv
import typer
import wandb
from transformers import TrainerCallback, PreTrainedTokenizer, AutoTokenizer
import numpy as np
from typing import Callable, Type, List, Any
import asyncio
from ..rm.utils import score_title
from ..utils import pull_data, cache, prompt_for_title
from panza import limit_concurrency
from abc import ABC, abstractmethod

load_dotenv()


def filter_on_length(data: Dataset, max_length: int, tokenizer_name: str) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def check_length(x):
        return (
            len(
                tokenizer.apply_chat_template(
                    x["prompt"], tokenize=True, add_generation_prompt=True
                )
            )
            <= max_length
        )

    len_before = len(data)
    data = data.filter(check_length)
    print(f"Samples before: {len_before}, samples after: {len(data)}")
    return data


@cache.cache()
async def load_data(
    split: str = "train",
    max_items: int = 10,
    min_score: int = 20,
    max_length: int = 8192,
    tokenizer_name: str = "unsloth/Qwen2.5-14B-Instruct",
) -> Dataset:
    data = pull_data(split=split, max_items=max_items, min_score=min_score)
    data = data.map(
        lambda x: {
            "prompt": prompt_for_title(x["scraped_body"]),
            "row": x,
        }
    )
    return filter_on_length(data, max_length, tokenizer_name)


class ValidationCallback(TrainerCallback):
    def __init__(
        self,
        val_dataset: Dataset,
        tokenizer: PreTrainedTokenizer,
        max_completion_length: int,
        reward_func: Callable,
        val_generations_to_log_to_wandb: int,
        eval_steps: int,
    ):
        self.val_dataset = val_dataset
        self.val_generations_to_log_to_wandb = val_generations_to_log_to_wandb
        self.validation_table = None
        self.inputs = tokenizer.apply_chat_template(
            self.val_dataset["prompt"], tokenize=False, add_generation_prompt=True
        )
        self.max_completion_length = max_completion_length
        self.reward_func = reward_func
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, model: FastLanguageModel, **kwargs):
        if state.global_step % self.eval_steps != 0:
            return control

        print(f"Model type: {type(model)}")
        # print(f"Model is: {model}")

        # You might think that `model.fast_generate` on a LoRA model would run generation on the actual LoRA model, but
        # you would be wrong, it runs generation on the base model! But if we save the LoRA weights and then load them,
        # we can get the correct behavior.
        model.save_lora("val_model_lora")  # type: ignore

        outputs = model.fast_generate(  # type: ignore
            self.inputs,
            sampling_params=SamplingParams(max_tokens=self.max_completion_length),
            lora_request=model.load_lora("val_model_lora"),  # type: ignore
        )
        outputs = [o.outputs[0].text for o in outputs]

        scores = self.reward_func(
            self.val_dataset["prompt"],
            [
                [{"content": o}] for o in outputs
            ],  # adapt to the shape the reward function expects
            self.val_dataset["row"],
        )
        mean_score = np.mean(scores)
        p5_score = np.percentile(scores, 5)
        median_score = np.percentile(scores, 50)
        p95_score = np.percentile(scores, 95)
        wandb.log(
            {
                "val/reward_mean": mean_score,
                "val/reward_p5": p5_score,
                "val/reward_median": median_score,
                "val/reward_p95": p95_score,
            },
            step=state.global_step,
        )

        generations_to_log = self.val_generations_to_log_to_wandb
        if generations_to_log == 0:
            return
        # Create tuples of (input, output, score) and sort by input text
        samples = [x for x in zip(self.inputs, outputs, scores)][:generations_to_log]
        # Create column names for all samples
        columns = ["step"] + sum(
            [
                [f"input_{i + 1}", f"output_{i + 1}", f"score_{i + 1}"]
                for i in range(len(samples))
            ],
            [],
        )

        if self.validation_table is None:
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = [state.global_step]
        for sample in samples:
            row_data.extend(sample)  # type: ignore
        new_table.add_data(*row_data)

        # Log the table and update reference
        wandb.log({"val/generations": new_table}, step=state.global_step)
        self.validation_table = new_table


class TaskConfig(ABC):
    @classmethod
    @abstractmethod
    async def load_data(
        cls, split: str, max_items: int, max_length: int, tokenizer_name: str
    ):
        pass

    @classmethod
    @abstractmethod
    def reward_funcs(cls) -> List[RewardFunc]:
        """Return a list of reward functions, each with the signature (prompts, completions, row) -> list[float]"""
        pass


class TitlesTaskConfig(TaskConfig):
    @classmethod
    async def load_data(
        cls,
        split: str,
        max_items: int,
        max_length: int,
        tokenizer_name: str,
    ):
        return await load_data(
            split=split,
            max_items=max_items,
            min_score=20,
            max_length=max_length,
            tokenizer_name=tokenizer_name,
        )

    @classmethod
    def reward_funcs(cls) -> List[RewardFunc]:
        def title_reward(prompts, completions, row) -> list[float]:
            responses = [completion[0]["content"] for completion in completions]
            updated_rows = [
                {**r, "title": response} for r, response in zip(row, responses)
            ]

            @limit_concurrency(10)
            async def score_title_async(r):
                return await score_title(r, "rm")

            async def run_scores():
                # await score_title.bust_cache()
                return await asyncio.gather(
                    *[score_title_async(r) for r in updated_rows]
                )

            return asyncio.run(run_scores())

        return [title_reward]  # type: ignore


class DebugTaskConfig(TaskConfig):
    @classmethod
    async def load_data(
        cls, split: str, max_items: int, max_length: int, tokenizer_name: str
    ):
        return Dataset.from_list(
            [
                {
                    "prompt": [
                        {
                            "role": "user",
                            "content": "Return a random number between 1 and 3. Literally just a single number, no other text.",
                        }
                    ],
                    "row": {},
                }
                for _ in range(max_items)
            ]
        )

    @classmethod
    def reward_funcs(cls) -> List[RewardFunc]:
        def debug_reward(prompts, completions, row, **kwargs):
            scores = []
            for comp in completions:
                try:
                    response = comp[0]["content"].strip()
                    score = 1 if response == "3" else 0
                except Exception:
                    score = 0
                scores.append(score)
            return scores

        return [debug_reward]  # type: ignore


def main(
    run_name: str = typer.Option(
        __file__.split("/")[-1].replace(".py", ""), help="Run name"
    ),
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
    max_steps: int = typer.Option(-1, help="Maximum training steps"),
    save_steps: int = typer.Option(250, help="Steps interval for saving"),
    max_grad_norm: float = typer.Option(0.1, help="Maximum gradient norm"),
    output_dir: str = typer.Option("outputs", help="Output directory"),
    training_dataset_size: int = typer.Option(
        10, help="Number of training samples to load"
    ),
    val_set_size: int = typer.Option(5, help="Number of validation samples to use"),
    val_samples_to_log: int = typer.Option(
        5, help="Number of validation samples to log to wandb"
    ),
    eval_steps: int = typer.Option(1, help="Evaluate every N training steps"),
    num_epochs: int = typer.Option(1, help="Number of training epochs"),
    task: str = typer.Option("titles", help="Task to run: 'titles' or 'debug'"),
) -> None:
    wandb.init(
        project="hn_title_debug" if task == "debug" else "hn_title_generation",
        name=run_name,
    )

    task_cls: Type[TaskConfig]
    if task == "debug":
        task_cls = DebugTaskConfig
    elif task == "titles":
        task_cls = TitlesTaskConfig
    else:
        raise ValueError(f"Unknown task: {task}. Valid choices are: 'titles', 'debug'.")

    dataset = asyncio.run(
        task_cls.load_data(
            split="train",
            max_items=training_dataset_size,
            max_length=max_prompt_length,
            tokenizer_name=base_model,
        )
    )
    val_dataset = asyncio.run(
        task_cls.load_data(
            split="val",
            max_items=val_set_size,
            max_length=max_prompt_length,
            tokenizer_name=base_model,
        )
    )

    print(f"Training dataset size: {len(dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Load model and tokenizer using the provided hyperparameters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_prompt_length + max_completion_length,
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
        use_gradient_checkpointing="unsloth",  # type: ignore
        random_state=3407,
    )

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
        num_train_epochs=num_epochs,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=task_cls.reward_funcs(),
        args=training_args,
        train_dataset=dataset,
    )
    validation_callback = ValidationCallback(
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        max_completion_length=max_completion_length,
        reward_funcs=task_cls.reward_funcs(),
        val_generations_to_log_to_wandb=val_samples_to_log,
        eval_steps=eval_steps,
    )
    trainer.add_callback(validation_callback)
    trainer.train()

    model.save_lora(run_name)


typer.run(main)
