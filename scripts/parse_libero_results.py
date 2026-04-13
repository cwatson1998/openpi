"""Parse LIBERO eval results and report per-task success rates against a train/eval split file."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/libero/videos/results.json"),
        help="Path to results.json from the eval run.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("data/libero/object_7train_3test.json"),
        help="Path to the task-split JSON file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/libero/videos/results_report.md"),
        help="Where to write the Markdown report.",
    )
    args = parser.parse_args()

    results = json.loads(args.results.read_text())
    split = json.loads(args.split.read_text())

    # Build a lookup: plain-English task description -> full task result entry.
    # Prefer matching by description string; fall back to index-based matching via
    # selected_tasks when task_results uses a different format (e.g. logic strings).
    raw_task_results: list[dict] = results["task_results"]
    selected_tasks: list[str] = results.get("selected_tasks", [])

    task_by_desc: dict[str, dict] = {}
    for entry in raw_task_results:
        task_by_desc[entry["task_description"]] = entry

    # If the descriptions don't match the split instructions, remap via selected_tasks index
    train_instrs: list[str] = split["train_task_instructions"]
    eval_instrs: list[str] = split["eval_task_instructions"]
    all_instrs = train_instrs + eval_instrs

    if selected_tasks and not any(t in task_by_desc for t in all_instrs):
        remapped: dict[str, dict] = {}
        for i, plain in enumerate(selected_tasks):
            if i < len(raw_task_results):
                remapped[plain] = raw_task_results[i]
        task_by_desc = remapped
    n_trials: int = results.get("num_trials_per_task", 0)
    suite: str = results.get("task_suite_name", "unknown")
    generated_at: str = results.get("generated_at", "unknown")

    def bar(rate: float) -> str:
        filled = round(rate * 20)
        return "█" * filled + "░" * (20 - filled)

    def section_avg(instructions: list[str]) -> float | None:
        rates = [task_by_desc[t]["success_rate"] for t in instructions if t in task_by_desc]
        return sum(rates) / len(rates) if rates else None

    lines: list[str] = []

    lines += [
        f"# LIBERO Eval Results — `{suite}`",
        "",
        f"**Generated:** {generated_at}  ",
        f"**Trials per task:** {n_trials}  ",
        f"**Overall:** {results['total_success_rate']*100:.1f}%"
        f"  ({results['total_successes']}/{results['total_episodes']})",
        "",
    ]

    for section_title, instructions in [("Train Tasks", train_instrs), ("Eval Tasks", eval_instrs)]:
        avg = section_avg(instructions)
        avg_str = f"{avg*100:.1f}%" if avg is not None else "N/A"

        lines += [
            f"---",
            "",
            f"## {section_title}",
            "",
            f"| # | Task | Success Rate | Pass / Fail | Progress |",
            f"|---|------|:------------:|:-----------:|----------|",
        ]

        for idx, instr in enumerate(instructions, 1):
            task = task_by_desc.get(instr)
            if task is None:
                lines.append(f"| {idx} | {instr} | N/A | — | — |")
                continue

            rate = task["success_rate"]
            successes = task["successes"]
            episodes = task["episodes"]
            failures = episodes - successes
            pass_fail = f"{successes} ✓ / {failures} ✗"
            progress = bar(rate)
            lines.append(f"| {idx} | {instr} | **{rate*100:.0f}%** | {pass_fail} | `{progress}` |")

        lines += [
            f"| | **Section average** | **{avg_str}** | | |",
            "",
        ]

        # Per-task episode breakdown
        lines += [
            f"### Episode-level detail",
            "",
        ]
        for idx, instr in enumerate(instructions, 1):
            task = task_by_desc.get(instr)
            if task is None:
                continue
            rate = task["success_rate"]
            lines += [
                f"#### {idx}. {instr}  `{rate*100:.0f}%`",
                "",
                "| Episode | Result | Steps |",
                "|:-------:|--------|------:|",
            ]
            for ep in task["episode_results"]:
                icon = "✓" if ep["success"] else "✗"
                lines.append(f"| {ep['episode_index']} | {icon} | {ep['steps_taken']} |")
            lines.append("")

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Report written to: {out_path}")

    # Also print a compact summary to stdout
    print(f"\nSuite : {suite}")
    print(f"Overall: {results['total_success_rate']*100:.1f}%  ({results['total_successes']}/{results['total_episodes']})")
    for section_title, instructions in [("TRAIN", train_instrs), ("EVAL", eval_instrs)]:
        avg = section_avg(instructions)
        avg_str = f"{avg*100:.1f}%" if avg is not None else "N/A"
        print(f"\n{section_title} tasks  (avg {avg_str}):")
        for instr in instructions:
            task = task_by_desc.get(instr)
            rate = task["success_rate"] if task else None
            pct = f"{rate*100:5.1f}%" if rate is not None else "  N/A "
            print(f"  {pct}  {instr}")


if __name__ == "__main__":
    main()
