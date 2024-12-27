# uv run modal run --detach scraped_stories.rm.model1

from .modal_app import app
import modal
import logging

s3_bucket_name = "[placeholder]"

if modal.is_local():
    import os
    from dotenv import load_dotenv

    load_dotenv()

    s3_bucket_name = os.getenv("REMOTE_BUCKET", "[placeholder]")
    logging.info(f"Using S3 bucket: {s3_bucket_name}")


@app.function(
    secrets=[modal.Secret.from_dotenv(".env")],
    gpu="H100",
    memory=32768 * 2,
    timeout=3600 * 24,
    volumes={
        "/remote": modal.CloudBucketMount(
            bucket_name=s3_bucket_name,
            secret=modal.Secret.from_dotenv(".env"),
            read_only=False,
        )
    },
)
def main():
    import os
    import polars as pl
    from .training_helpers import (
        run_final_inference_and_report_metrics,
        load_model,
    )

    # Configuration
    model_dir = f"/remote/rm/models/model1"

    logging.info(f"Loading model from {model_dir}")
    model = load_model(model_dir)

    logging.info("Loading dataset...")
    df = pl.read_parquet(
        f"s3://{os.getenv('REMOTE_BUCKET')}/scraped-stories-with-datetime.parquet"
    )

    logging.info("Running final inference and reporting metrics...")
    run_final_inference_and_report_metrics(model, df, model_dir)

    logging.info("Inference complete")


@app.local_entrypoint()
def main_local():
    print("Running main locally")
    main.remote()
