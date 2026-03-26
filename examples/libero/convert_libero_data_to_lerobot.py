"""
Convert LIBERO RLDS data to LeRobot format.

Examples:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir data/libero/raw --download

uv run examples/libero/convert_libero_data_to_lerobot.py \
    --data_dir data/libero/raw \
    --repo_name local/libero_object \
    --suite_names libero_object \
    --task_suite_name libero_object \
    --task_indices 0 1 2 3 4 5 6
"""

from collections import Counter
from collections.abc import Sequence
import json
import pathlib
import shutil

from huggingface_hub import snapshot_download
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

from openpi.training import libero as libero_utils


def _download_raw_datasets(data_dir: pathlib.Path, raw_dataset_names: Sequence[str]) -> None:
    allow_patterns = [f"{raw_dataset_name}/*" for raw_dataset_name in raw_dataset_names]
    snapshot_download(
        repo_id="openvla/modified_libero_rlds",
        repo_type="dataset",
        local_dir=str(data_dir),
        allow_patterns=allow_patterns,
    )


def _write_subset_metadata(
    output_path: pathlib.Path,
    *,
    selected_tasks: Sequence[str],
    suite_names: Sequence[str],
    task_counts: Counter[str],
    skipped_task_counts: Counter[str],
) -> None:
    metadata = {
        "suite_names": list(suite_names),
        "selected_task_instructions": list(selected_tasks),
        "selected_episode_counts": dict(sorted(task_counts.items())),
        "skipped_episode_counts": dict(sorted(skipped_task_counts.items())),
    }
    metadata_path = output_path / "meta" / "libero_subset.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main(
    data_dir: str,
    *,
    repo_name: str = "your_hf_username/libero",
    suite_names: Sequence[str] = tuple(libero_utils.LIBERO_RAW_DATASETS),
    download: bool = False,
    task_suite_name: str | None = None,
    task_indices: Sequence[int] = (),
    task_names: Sequence[str] = (),
    task_split_file: str | None = None,
    task_split: str = "train",
    push_to_hub: bool = False,
):
    data_dir_path = pathlib.Path(data_dir).expanduser().resolve()
    raw_dataset_names = [libero_utils.LIBERO_RAW_DATASETS[suite_name] for suite_name in suite_names]
    selected_tasks = libero_utils.resolve_task_filters(
        task_suite_name=task_suite_name,
        task_indices=task_indices,
        task_names=task_names,
        task_split_file=task_split_file,
        task_split=task_split,
    )
    selected_task_set = {task.casefold() for task in selected_tasks}

    if download:
        _download_raw_datasets(data_dir_path, raw_dataset_names)

    output_path = LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    written_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    for raw_dataset_name in raw_dataset_names:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=str(data_dir_path), split="train")
        for episode in raw_dataset:
            steps = list(episode["steps"].as_numpy_iterator())
            task_instruction = steps[-1]["language_instruction"].decode()

            if selected_task_set and task_instruction.casefold() not in selected_task_set:
                skipped_counts[task_instruction] += 1
                continue

            for step in steps:
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                    }
                )
            dataset.save_episode(task=task_instruction)
            written_counts[task_instruction] += 1

    if not written_counts:
        raise ValueError("No LIBERO episodes matched the selected task filter.")

    dataset.consolidate(run_compute_stats=False)
    _write_subset_metadata(
        output_path,
        selected_tasks=selected_tasks,
        suite_names=suite_names,
        task_counts=written_counts,
        skipped_task_counts=skipped_counts,
    )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )

    print(f"Wrote {sum(written_counts.values())} episodes to {output_path}")
    for task_name, count in written_counts.most_common():
        print(f"  {count:3d}  {task_name}")


if __name__ == "__main__":
    tyro.cli(main)
