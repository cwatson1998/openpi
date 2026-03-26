import argparse
import pathlib

from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset to a local directory.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, e.g. org/name")
    parser.add_argument("--local-dir", required=True, help="Destination directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = pathlib.Path(args.local_dir).expanduser().resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    files = sorted(api.list_repo_files(repo_id=args.repo_id, repo_type="dataset"))
    total = len(files)

    print(f"Starting download of {total} files from {args.repo_id} into {local_dir}", flush=True)
    for index, filename in enumerate(files, start=1):
        print(f"[{index}/{total}] {filename}", flush=True)
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
        )

    print(f"Download complete: {local_dir}", flush=True)


if __name__ == "__main__":
    main()
