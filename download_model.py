"""Download the official pretrained weights from Hugging Face."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def ensure_model(repo: str, filename: str, target_name: str, destination: Path, force: bool = False) -> Path:
    target = destination / target_name
    if target.exists() and target.stat().st_size > 0 and not force:
        print(f"Using cached {target}")
        return target

    cached = hf_hub_download(repo_id=repo, filename=filename)
    target.write_bytes(Path(cached).read_bytes())
    print(f"Downloaded {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download cached model files")
    args = parser.parse_args()

    destination = Path(__file__).resolve().parent / "models"
    destination.mkdir(exist_ok=True)
    files = [
        ("Piyu12/fruit-veg-freshness-mobilenetv2", "freshness_mobilenetv2_state_dict.pth", "freshness_mobilenetv2_state_dict.pth"),
        ("bhumong/fruit-classifier-efficientnet-b0", "fruit_classifier_efficientnet_b0_best.pth", "produce_efficientnet_b0.pth"),
        ("bhumong/fruit-classifier-efficientnet-b0", "config.json", "produce_config.json"),
    ]
    for repo, filename, target_name in files:
        ensure_model(repo, filename, target_name, destination, force=args.force)


if __name__ == "__main__":
    main()
