import io
import subprocess
import tempfile
import os
import hashlib


def update_image():
    # Create hash of dockerfile contents
    dockerfile_hash = hashlib.sha256(dockerfile.encode()).hexdigest()[:12]
    image_name = f"ghcr.io/openpipe/scraped-stories-rm:{dockerfile_hash}"

    print(f"Building image {image_name}")

    # Create temporary directory and write dockerfile there
    with tempfile.TemporaryDirectory() as tmpdir:
        dockerfile_path = os.path.join(tmpdir, "Dockerfile")
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        subprocess.run(
            ["docker", "build", "--platform", "linux/amd64", "-t", image_name, tmpdir],
            check=True,
        )

    print(f"Pushing image {image_name}")
    subprocess.run(["docker", "push", image_name], check=True)

    print(f"Image {image_name} updated")

    return image_name


if __name__ == "__main__":
    update_image()
