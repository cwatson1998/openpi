from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib

from openpi.training import libero as libero_utils


def _default_bddl_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3] / "third_party/libero/libero/libero/bddl_files"


def _default_lerobot_cache_root() -> pathlib.Path:
    return pathlib.Path.home() / ".cache/huggingface/lerobot"


def _normalize_task_name(task_name: str) -> str:
    return " ".join(task_name.strip().replace("_", " ").split()).casefold()


def _extract_balanced_clause(text: str, clause_name: str) -> str:
    marker = f"(:{clause_name}"
    start = text.find(marker)
    if start == -1:
        raise ValueError(f"Clause not found: {marker}")

    depth = 0
    end = None
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError(f"Unbalanced parentheses while parsing {marker}")
    return text[start:end]


def extract_bddl_goal(bddl_text: str) -> str:
    goal_clause = _extract_balanced_clause(bddl_text, "goal")
    body = goal_clause[len("(:goal") : -1].strip()
    return " ".join(body.split())


@dataclasses.dataclass(frozen=True)
class LiberoLogicTaskDescription:
    task_id: str
    task_instruction: str
    bddl_file: str
    goal_description: str
    logic_task_description: str


@dataclasses.dataclass(frozen=True)
class LiberoLogicDatasetDescription:
    dataset_name: str
    suite_name: str
    tasks: list[LiberoLogicTaskDescription]


def load_task_descriptions(path: str | pathlib.Path) -> LiberoLogicDatasetDescription:
    raw = json.loads(pathlib.Path(path).read_text())
    return LiberoLogicDatasetDescription(
        dataset_name=raw["dataset_name"],
        suite_name=raw["suite_name"],
        tasks=[LiberoLogicTaskDescription(**task) for task in raw["tasks"]],
    )


def build_task_prompt_map_from_dataset_tasks(
    dataset_tasks: dict[int, str],
    task_description_path: str | pathlib.Path,
) -> dict[int, str]:
    description_file = load_task_descriptions(task_description_path)
    by_instruction = {
        _normalize_task_name(task.task_instruction): task.logic_task_description for task in description_file.tasks
    }

    prompt_map = {}
    missing = []
    for task_index, task_instruction in dataset_tasks.items():
        normalized = _normalize_task_name(task_instruction)
        if normalized not in by_instruction:
            missing.append(task_instruction)
            continue
        prompt_map[int(task_index)] = by_instruction[normalized]

    if missing:
        raise ValueError(
            "Task description file does not cover all dataset tasks. Missing descriptions for:\n"
            + "\n".join(f"  - {task}" for task in missing)
        )

    return prompt_map


def infer_libero_suite_name(
    dataset_name: str,
    *,
    task_split_file: str | None = None,
    lerobot_cache_root: pathlib.Path | None = None,
) -> str:
    if task_split_file is not None:
        return libero_utils.load_task_split(task_split_file).suite_name

    if dataset_name in libero_utils.LIBERO_TASK_IDS:
        return dataset_name

    for suite_name, raw_dataset_name in libero_utils.LIBERO_RAW_DATASETS.items():
        if dataset_name == raw_dataset_name:
            return suite_name

    maybe_dataset_root = resolve_lerobot_dataset_root(dataset_name, lerobot_cache_root=lerobot_cache_root)
    if maybe_dataset_root is not None:
        subset_path = maybe_dataset_root / "meta/libero_subset.json"
        if subset_path.exists():
            subset = json.loads(subset_path.read_text())
            suite_names = subset.get("suite_names", [])
            if len(suite_names) == 1:
                return str(suite_names[0])

    normalized = dataset_name.casefold()
    matches = [suite_name for suite_name in libero_utils.LIBERO_TASK_IDS if suite_name in normalized]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Could not infer LIBERO suite from dataset name: {dataset_name}")
    raise ValueError(f"Dataset name is ambiguous across LIBERO suites: {dataset_name} -> {matches}")


def resolve_lerobot_dataset_root(
    dataset_name: str,
    *,
    lerobot_cache_root: pathlib.Path | None = None,
) -> pathlib.Path | None:
    candidate = pathlib.Path(dataset_name).expanduser()
    if candidate.exists():
        return candidate

    if dataset_name.startswith("local/"):
        cache_root = lerobot_cache_root or _default_lerobot_cache_root()
        local_name = dataset_name.split("/", 1)[1]
        candidate = cache_root / "local" / local_name
        if candidate.exists():
            return candidate

    return None


def _resolve_selected_task_instructions(
    dataset_name: str,
    suite_name: str,
    *,
    task_split_file: str | None = None,
    task_split: str = "train",
    lerobot_cache_root: pathlib.Path | None = None,
) -> tuple[str, ...]:
    if task_split_file is not None:
        return libero_utils.resolve_task_filters(task_split_file=task_split_file, task_split=task_split)

    dataset_root = resolve_lerobot_dataset_root(dataset_name, lerobot_cache_root=lerobot_cache_root)
    if dataset_root is not None:
        subset_path = dataset_root / "meta/libero_subset.json"
        if subset_path.exists():
            subset = json.loads(subset_path.read_text())
            selected = subset.get("selected_task_instructions", [])
            if selected:
                return tuple(str(task) for task in selected)

    return libero_utils.get_libero_task_instructions(suite_name)


def build_libero_logic_task_descriptions(
    dataset_name: str,
    *,
    task_split_file: str | None = None,
    task_split: str = "train",
    bddl_root: pathlib.Path | None = None,
    lerobot_cache_root: pathlib.Path | None = None,
) -> LiberoLogicDatasetDescription:
    suite_name = infer_libero_suite_name(
        dataset_name,
        task_split_file=task_split_file,
        lerobot_cache_root=lerobot_cache_root,
    )
    selected_tasks = _resolve_selected_task_instructions(
        dataset_name,
        suite_name,
        task_split_file=task_split_file,
        task_split=task_split,
        lerobot_cache_root=lerobot_cache_root,
    )

    bddl_root = bddl_root or _default_bddl_root()
    task_table = libero_utils.get_libero_task_table(suite_name)
    by_instruction = {_normalize_task_name(instruction): task_id for _, task_id, instruction in task_table}

    tasks = []
    for task_instruction in selected_tasks:
        normalized = _normalize_task_name(task_instruction)
        if normalized not in by_instruction:
            raise ValueError(f"Task instruction not found in suite {suite_name}: {task_instruction}")

        task_id = by_instruction[normalized]
        bddl_path = bddl_root / suite_name / f"{task_id}.bddl"
        if not bddl_path.exists():
            raise FileNotFoundError(f"BDDL file not found: {bddl_path}")

        goal_description = extract_bddl_goal(bddl_path.read_text())
        tasks.append(
            LiberoLogicTaskDescription(
                task_id=task_id,
                task_instruction=task_instruction,
                bddl_file=str(bddl_path),
                goal_description=goal_description,
                logic_task_description=goal_description,
            )
        )

    return LiberoLogicDatasetDescription(dataset_name=dataset_name, suite_name=suite_name, tasks=tasks)


@dataclasses.dataclass(frozen=True)
class Args:
    dataset_name: str
    task_split_file: str | None = None
    task_split: str = "train"
    bddl_root: str | None = None
    out: str | None = None


def main(args: Args) -> None:
    result = build_libero_logic_task_descriptions(
        args.dataset_name,
        task_split_file=args.task_split_file,
        task_split=args.task_split,
        bddl_root=pathlib.Path(args.bddl_root) if args.bddl_root else None,
    )
    payload = dataclasses.asdict(result)
    serialized = json.dumps(payload, indent=2) + "\n"

    if args.out is None:
        print(serialized, end="")
        return

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(serialized)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("--task-split-file")
    parser.add_argument("--task-split", default="train")
    parser.add_argument("--bddl-root")
    parser.add_argument("--out")
    parsed = parser.parse_args()
    main(
        Args(
            dataset_name=parsed.dataset_name,
            task_split_file=parsed.task_split_file,
            task_split=parsed.task_split,
            bddl_root=parsed.bddl_root,
            out=parsed.out,
        )
    )
