from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_serializer
import os
import torch
import uvicorn
from datetime import datetime
from typing import Optional
import logging
import time

from dotenv import load_dotenv  # load environment variables from .env
import s3fs
import hashlib

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from rm.utils import serialize_story, ScoreRequest
from liger_kernel.transformers import _apply_liger_kernel_to_instance

# Load .env file to retrieve S3 credentials and other environment variables.
load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScoreResponse(BaseModel):
    score: float
    version: str


# Read the checkpoint path for the PEFT model from the environment.
peft_model_path = os.getenv("REWARD_MODEL_NAME", "default_model")
print(f"Initializing reward model with PEFT checkpoint: {peft_model_path}")

# If the checkpoint path is an S3 URI, create a local store directory based on its hash.
if peft_model_path.startswith("s3://"):
    # Create a hash of the S3 path string.
    hash_string = hashlib.sha256(peft_model_path.encode("utf-8")).hexdigest()
    local_checkpoint_dir = f"/tmp/peft_checkpoint_{hash_string}"

    # Check if the checkpoint already exists locally.
    if not os.path.exists(local_checkpoint_dir):
        print(
            f"Local checkpoint not found at {local_checkpoint_dir}. Downloading from S3: {peft_model_path}"
        )
        fs = s3fs.S3FileSystem()
        fs.get(peft_model_path, local_checkpoint_dir, recursive=True)
    else:
        print(f"Local checkpoint found at {local_checkpoint_dir}. Skipping download.")

    # Update the model path to point to the local directory.
    peft_model_path = local_checkpoint_dir

# Load the PEFT configuration to retrieve the base model name.
peft_config = PeftConfig.from_pretrained(peft_model_path)
base_model_name = peft_config.base_model_name_or_path
print(f"Base model name from PEFT config: {base_model_name}")

# Load the tokenizer from the base model.
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the base model using the reward model pattern (from inference.py).
print("Loading base model for reward model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name, num_labels=1, device_map="auto", torch_dtype=torch.bfloat16
)

# Load the PEFT adapter from the checkpoint and merge the adapter weights.
print("Loading and merging PEFT adapter...")
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)
model = peft_model.merge_and_unload()  # Merge PEFT weights into the base model.
_apply_liger_kernel_to_instance(model)
model.eval()


@app.post("/score")
async def get_score(request: ScoreRequest) -> ScoreResponse:
    start_time = time.time()  # Added to record request start time
    try:
        # Convert Pydantic model to dict and serialize the story
        story_dict = request.model_dump()
        story_dict["time"] = datetime.fromisoformat(story_dict["time"])
        serialized_text = serialize_story(story_dict)

        # Tokenize the serialized text
        inputs = tokenizer(
            serialized_text, return_tensors="pt", padding=True, truncation=True
        )
        token_count = inputs["input_ids"].shape[-1]  # Log the number of input tokens
        # Move tensors to the same device as the model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        # Assume the merged model returns logits with shape [1,1] and use that as the score
        score = outputs.logits[0][0].item()

        duration = time.time() - start_time
        logger.info(
            f"Request completed: input tokens: {token_count}, duration: {duration:.2f}s"
        )
        return ScoreResponse(score=score, version="0.1")
    except Exception as e:
        duration = time.time() - start_time
        logger.info(f"Request failed: duration: {duration:.2f}s")
        logger.exception("Error in /score endpoint")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    print("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=80)
