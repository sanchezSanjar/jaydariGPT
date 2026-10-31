"""One-time online setup: download TinyLlama so chat can run offline afterwards."""
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
target = Path("models/TinyLlama-1.1B-Chat-v1.0")
target.mkdir(parents=True, exist_ok=True)
print(f"Downloading {repo_id} to {target} ...")
snapshot_download(
    repo_id=repo_id,
    local_dir=str(target),
)
print("\nBase model downloaded. JaydariGPT chat can now run without internet.")
