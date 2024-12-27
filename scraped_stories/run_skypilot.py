import sky
import dotenv
import os
import asyncio
from panza import limit_concurrency
import hashlib
import tempfile
import subprocess

dotenv.load_dotenv()


@limit_concurrency(n=1, derive_key=lambda dockerfile_contents: dockerfile_contents)
async def create_image(dockerfile_contents: str):
    # Create hash of dockerfile contents
    dockerfile_hash = hashlib.sha256(dockerfile_contents.encode()).hexdigest()[:12]
    image_name = f"ghcr.io/openpipe/skypilot-images:{dockerfile_hash}"

    # Check if image already exists
    process = await asyncio.create_subprocess_exec(
        "docker",
        "manifest",
        "inspect",
        image_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await process.communicate()

    # If return code is 0, image exists, so we can return early
    if process.returncode == 0:
        return image_name

    print(f"Building image {image_name}")

    # Create temporary directory and write dockerfile there
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile_contents)

        process = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            image_name,
            tmpdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode or 1, "docker build")

    print(f"Pushing image {image_name}")
    process = await asyncio.create_subprocess_exec(
        "docker",
        "push",
        image_name,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process.communicate()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode or 1, "docker push")

    print(f"Image {image_name} updated")

    return image_name


async def run_skypilot(
    script_path: str,
    env_vars: dict[str, str] | None = None,
    dockerfile_contents: str = "FROM python:3.12",
):
    docker_image = await create_image(dockerfile_contents)
    env_vars = {**dotenv.dotenv_values(), **(env_vars or {})}  # type: ignore

    task = sky.Task(
        # run="pwd && ls",
        run=f"python {script_path}",
        envs=env_vars,
        workdir=os.path.dirname(os.path.abspath(__file__)),
    )

    task.set_resources(
        sky.Resources(
            image_id=docker_image, cloud=sky.RunPod(), accelerators="RTX4090:1"
        )
    )

    # Wrap the blocking sky.launch call in an executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None, lambda: sky.launch(task, down=True, stream_logs=True)
    )


def is_local():
    return "SKYPILOT_TASK_ID" not in os.environ


if __name__ == "__main__":
    dockerfile_contents = """
FROM python:3.12
RUN pip install polars
"""

    asyncio.run(run_skypilot("rm/test.py", dockerfile_contents=dockerfile_contents))
