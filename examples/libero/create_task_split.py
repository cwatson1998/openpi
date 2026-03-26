"""Create a reusable LIBERO train/eval task split JSON file."""

from collections.abc import Sequence
import pathlib

import tyro

from openpi.training import libero as libero_utils


def main(
    output: str,
    suite_name: str,
    *,
    train_task_indices: Sequence[int] = (),
    train_task_names: Sequence[str] = (),
    eval_task_indices: Sequence[int] = (),
    eval_task_names: Sequence[str] = (),
    use_remaining_tasks_for_eval: bool = True,
) -> None:
    split = libero_utils.write_task_split(
        output,
        suite_name=suite_name,
        train_task_indices=train_task_indices,
        train_task_names=train_task_names,
        eval_task_indices=eval_task_indices,
        eval_task_names=eval_task_names,
        use_remaining_tasks_for_eval=use_remaining_tasks_for_eval,
    )

    print(f"Wrote task split to {pathlib.Path(output).resolve()}")
    print("Available tasks:")
    train_set = {task.casefold() for task in split.train_task_instructions}
    eval_set = {task.casefold() for task in split.eval_task_instructions}
    for task_index, _, task_instruction in libero_utils.get_libero_task_table(suite_name):
        split_name = (
            "train"
            if task_instruction.casefold() in train_set
            else "eval" if task_instruction.casefold() in eval_set else "-"
        )
        print(f"  [{split_name:5}] {task_index:2d}  {task_instruction}")


if __name__ == "__main__":
    tyro.cli(main)
