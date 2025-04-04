#!/usr/bin/env python
"""
Benchmark script to evaluate different LLMs against a reward model.
"""

import json
import os
import time
import asyncio
from typing import Dict, List, Tuple
import argparse
import statistics
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from panza import limit_concurrency
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from .utils import cache, prompt_for_title, pull_data
from .rm.utils import score_title

# Load environment variables
load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CURRENT_REWARD_MODEL = "model6"

# Models to benchmark via OpenRouter
MODELS = {
    "o3-mini": "openai/o3-mini",
    "gpt-4o": "openai/gpt-4o",
    "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-7-sonnet": "anthropic/claude-3.7-sonnet",
    "qwen-2.5-32b": "qwen/qwen2.5-32b-instruct",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
}


script_dir = os.path.dirname(os.path.abspath(__file__))

openrouter_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1"
)


@cache.cache()
@limit_concurrency(10)
async def call_openrouter(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 5000,
) -> Tuple[ChatCompletion, float]:
    """
    Make an async call to OpenRouter API using the AsyncOpenAI client.

    Args:
        model: The model identifier to use.
        messages: List of message objects with role and content.
        temperature: Sampling temperature (0-1).
        max_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (full API response as dict, elapsed time in seconds).
    """
    start_time = time.time()

    # Make the async API call using the client.
    response = await openrouter_client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
    )

    elapsed_time = time.time() - start_time
    return response, elapsed_time


async def process_item(item: Dict, model_name: str) -> Dict:
    """Process a single dataset item for benchmarking."""
    url = item["url"]
    body = item["scraped_body"]
    original_title = (
        item["original_title"] if "original_title" in item else item["title"]
    )
    # Get the original score from the dataset if available
    original_score = item.get("score", None)

    start_time = time.time()
    try:
        truncated_content = body[:4000] if body else ""
        messages = prompt_for_title(truncated_content)

        api_response, api_time = await call_openrouter(
            model_name, messages, max_tokens=5000
        )
        content = api_response.choices[0].message.content
        title = content.strip() if content is not None else ""

        # Strip quotes from the title if present
        if title.startswith('"') and title.endswith('"'):
            title = title[1:-1].strip()
        elif title.startswith("'") and title.endswith("'"):
            title = title[1:-1].strip()

        total_time = time.time() - start_time

        # Build and serialize the generated story with full metadata
        generated_story = {
            "by": item["by"],
            "url": url,
            "time": item["time"],
            "scraped_body": body,
            "title": title,
        }
        score = await score_title(generated_story, CURRENT_REWARD_MODEL)

        # Build and serialize the original story if available
        if original_title:
            original_story = {
                "by": item["by"],
                "url": url,
                "time": item["time"],
                "scraped_body": body,
                "title": original_title,
            }
            rm_score_original = await score_title(original_story, CURRENT_REWARD_MODEL)
        else:
            rm_score_original = None

        return {
            "item_id": item["id"],
            "generated_title": title,
            "original_title": original_title,
            "original_score": original_score,  # Score from the dataset
            "score": score,  # RM score for generated title
            "rm_score_original": rm_score_original,  # RM score for original title
            "api_time": api_time,
            "total_time": total_time,
            "full_response": api_response.model_dump(),
            "scraped_body": body,
        }
    except Exception as e:
        total_time = time.time() - start_time

        if original_title:
            original_story = {
                "by": item["by"],
                "url": url,
                "time": item["time"],
                "scraped_body": body,
                "title": original_title,
            }
            rm_score_original = await score_title(original_story, CURRENT_REWARD_MODEL)
        else:
            rm_score_original = None
        return {
            "item_id": item["id"],
            "generated_title": None,
            "original_title": original_title,
            "original_score": original_score,  # Score from the dataset
            "score": None,  # RM score for generated title
            "rm_score_original": rm_score_original,  # RM score for original title
            "error": str(e),
            "total_time": total_time,
            "api_time": None,
            "scraped_body": body,
            "full_response": None,
        }


async def benchmark_model(model_name: str, dataset: list[dict]) -> dict:
    """Benchmark a single model on the dataset asynchronously."""
    tasks = [asyncio.create_task(process_item(item, model_name)) for item in dataset]
    results = await tqdm.gather(
        *tasks, desc=f"Benchmarking {model_name}", total=len(tasks)
    )

    generation_times = [r["total_time"] for r in results if r["score"] is not None]
    api_times = [
        r["api_time"]
        for r in results
        if r["score"] is not None and r["api_time"] is not None
    ]

    valid_results = [r for r in results if r["score"] is not None]
    valid_rm_original_scores = [
        r["rm_score_original"] for r in results if r["rm_score_original"] is not None
    ]

    if valid_results:
        scores = [r["score"] for r in valid_results]
        avg_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        avg_total_time = statistics.mean(generation_times) if generation_times else 0
        avg_api_time = statistics.mean(api_times) if api_times else 0
    else:
        avg_score = median_score = avg_total_time = avg_api_time = 0

    avg_rm_original_score = (
        statistics.mean(valid_rm_original_scores) if valid_rm_original_scores else None
    )

    return {
        "model": model_name,
        "results": results,
        "metrics": {
            "avg_score": avg_score,
            "median_score": median_score,
            "avg_total_time": avg_total_time,
            "avg_api_time": avg_api_time,
            "avg_rm_original_score": avg_rm_original_score,
            "sample_count": len(results),
            "valid_sample_count": len(valid_results),
        },
    }


async def main():
    # await call_openrouter.bust_cache()
    # await score_title.bust_cache()
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs against a reward model"
    )
    parser.add_argument(
        "--output", type=str, default="benchmark_results.json", help="Output file path"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=MODELS.keys(),
        help="Models to benchmark",
    )
    parser.add_argument(
        "--max-samples", type=int, default=10, help="Maximum number of samples to test"
    )

    args = parser.parse_args()

    # Always load the validation dataset from HuggingFace.
    dataset: list[dict] = list(
        pull_data(split="val", max_items=args.max_samples, min_score=20)
    )  # type: ignore
    print(f"Loaded validation dataset with {len(dataset)} items")

    # Determine which models to benchmark.
    models_to_benchmark = args.models if args.models else list(MODELS.keys())

    # Run benchmarks concurrently for each model.
    tasks = [
        benchmark_model(MODELS[model_key], dataset) for model_key in models_to_benchmark
    ]
    results_list = await tqdm.gather(*tasks, desc="Benchmarking", total=len(tasks))

    # Group results by story (item_id)
    stories = {}
    for model_key, model_result in zip(models_to_benchmark, results_list):
        for res in model_result["results"]:
            story_id = res["item_id"]
            if story_id not in stories:
                stories[story_id] = {
                    "item_id": story_id,
                    "original_title": res["original_title"],
                    "original_score": res["original_score"],  # Score from the dataset
                    "rm_score_original": res[
                        "rm_score_original"
                    ],  # RM score for original title
                    "scraped_body": res["scraped_body"],
                    "model_results": {},
                }
            stories[story_id]["model_results"][model_key] = {
                "generated_title": res["generated_title"],
                "score": res["score"],  # RM score for generated title
                "api_time": res["api_time"],
                "total_time": res["total_time"],
                "full_response": res["full_response"],
            }

    # For simplicity, we do not compute overall metrics here.
    results = stories

    # Save full results by story.
    output_file = os.path.join(script_dir, args.output)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Full results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
