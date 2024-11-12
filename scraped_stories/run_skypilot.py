import sky
import dotenv
import os
from typing import Callable

dotenv.load_dotenv()


def run_skypilot(script_path: str, env_vars: dict[str, str] | None = None):
    env_vars = env_vars or dotenv.dotenv_values()  # type: ignore

    s3_bucket_name = os.getenv("REMOTE_BUCKET")
    task = sky.Task(
        run="pwd && ls",
        envs=env_vars,
        workdir=os.path.dirname(os.path.abspath(__file__)),
    )

    # task.set_resources(sky.Resources(cloud=sky.RunPod(), accelerators="H100:1"))
    task.set_resources(sky.Resources(cloud=sky.RunPod()))

    # task.set_storage_mounts({"/remote": sky.Storage(source=f"s3://{s3_bucket_name}")})

    sky.launch(task, cluster_name="scraped-stories-rm", down=True)


def is_local():
    return "SKYPILOT_TASK_ID" not in os.environ


if __name__ == "__main__":
    run_skypilot("scraped_stories/rm/model1.py")
