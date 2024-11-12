import modal

modal_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:23.10-py3",
    )
    .apt_install("git", "clang")
    .pip_install("uv")
    # Flash-attn takes forever to install so let's do it first. That also
    # requires installing its implicit dependencies.
    .run_commands(
        "pip install wheel packaging psutil torch==2.5.1 && pip install --no-build-isolation flash-attn==2.6.3"
    )
    .run_commands(
        "uv pip install --system --compile-bytecode "
        "transformers==4.45.2 "
        "datasets==3.1.0 "
        "wandb==0.18.5 "
        "trl==0.11.4 "
        "peft==0.13.2 "
        "liger-kernel==0.3.1 "
        "hf-transfer==0.1.8 "
        "polars==1.12.0 "
        "scikit-learn==1.5.2 "
        "scipy==1.14.1 "
        "pandas==2.2.3 "
    )
)

app = modal.App(image=modal_image)
