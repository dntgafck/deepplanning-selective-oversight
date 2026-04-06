from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET_REPO_ID = "Qwen/DeepPlanning"
DATASET_REPO_TYPE = "dataset"
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "deepplanning"


def download_file(candidates: list[str]) -> Path:
    last_error: Exception | None = None

    for candidate in candidates:
        try:
            return Path(
                hf_hub_download(
                    repo_id=DATASET_REPO_ID,
                    repo_type=DATASET_REPO_TYPE,
                    filename=candidate,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised against remote HF state
            last_error = exc

    candidate_list = ", ".join(candidates)
    raise RuntimeError(f"Unable to download any of: {candidate_list}") from last_error


def remove_if_exists(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def extract_tar(archive_path: Path, destination: Path, expected_output: Path) -> None:
    remove_if_exists(expected_output)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination)

    if not expected_output.exists():
        raise RuntimeError(
            f"Expected extracted directory is missing: {expected_output}"
        )


def extract_zip(archive_path: Path, destination: Path, expected_output: Path) -> None:
    remove_if_exists(expected_output)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(destination)

    if not expected_output.exists():
        raise RuntimeError(
            f"Expected extracted directory is missing: {expected_output}"
        )


def main() -> None:
    shopping_dir = DATA_ROOT / "shopping"
    travel_database_dir = DATA_ROOT / "travel" / "database"

    shopping_dir.mkdir(parents=True, exist_ok=True)
    travel_database_dir.mkdir(parents=True, exist_ok=True)

    shopping_archives = [
        {
            "candidates": [
                "shoppingplanning/database_zip/database_level1.tar.gz",
                "database_level1.tar.gz",
            ],
            "output": shopping_dir / "database_level1",
        },
        {
            "candidates": [
                "shoppingplanning/database_zip/database_level2.tar.gz",
                "database_level2.tar.gz",
            ],
            "output": shopping_dir / "database_level2",
        },
        {
            "candidates": [
                "shoppingplanning/database_zip/database_level3.tar.gz",
                "database_level3.tar.gz",
            ],
            "output": shopping_dir / "database_level3",
        },
    ]
    travel_archives = [
        {
            "candidates": [
                "travelplanning/database/database_zh.zip",
                "database_zh.zip",
            ],
            "output": travel_database_dir / "database_zh",
        },
        {
            "candidates": [
                "travelplanning/database/database_en.zip",
                "database_en.zip",
            ],
            "output": travel_database_dir / "database_en",
        },
    ]

    for item in shopping_archives:
        archive_path = download_file(item["candidates"])
        print(f"Extracting {archive_path.name} -> {item['output']}")
        extract_tar(archive_path, shopping_dir, item["output"])

    for item in travel_archives:
        archive_path = download_file(item["candidates"])
        print(f"Extracting {archive_path.name} -> {item['output']}")
        extract_zip(archive_path, travel_database_dir, item["output"])

    print("DeepPlanning benchmark data is ready.")


if __name__ == "__main__":
    main()
