import kagglehub
import os
from pathlib import Path

def download_dataset(dataset_name: str, target_dir: str = "data") -> Path:
    """
    Downloads a Kaggle dataset using kagglehub and returns the local path.
    Requires a valid Kaggle API key configured.
    """
    path = kagglehub.dataset_download(dataset_name)
    dest = Path(target_dir) / dataset_name.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)

    for f in Path(path).rglob("*"):
        if f.is_file():
            target_file = dest / f.relative_to(path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            if not target_file.exists():
                target_file.write_bytes(f.read_bytes())
    return dest
