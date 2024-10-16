import polars as pl
import requests
from tqdm import tqdm

# Read the Polars DataFrame
df = pl.read_parquet("./data/top_comments.parquet")

# df = df.sample(n=3005, seed=42)

print(f"Number of top comments: {len(df)}")


# sglang inference function
def run_inference_sglang(prompts: list[str]) -> list[float]:
    json_data = {
        "conv": prompts,
    }
    response = requests.post("http://127.0.0.1:30000/judge", json=json_data).json()
    return [x["embedding"][0] for x in response]


# Process comments in batches of 1000
batch_size = 1000

prompts = df["prompt"].to_list()
rewards = []

for batch in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
    batch_prompts = prompts[batch : batch + batch_size]
    batch_rewards = run_inference_sglang(batch_prompts)
    rewards.extend(batch_rewards)

# Add rewards to the DataFrame
df = df.with_columns(pl.Series("reward", rewards))

# Save the updated DataFrame
df.write_parquet("./data/top_comments_with_reward.parquet")

print("Processing complete. Results saved to top_comments_with_reward.parquet")
