import collections
import dataclasses
import json
import logging
import math
import pathlib
from datetime import datetime, timezone

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

from openpi.training import libero as libero_utils

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    task_indices: tuple[int, ...] = ()
    task_names: tuple[str, ...] = ()
    task_split_file: str | None = None
    task_split: str = "eval"
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    # Output directory for rollout videos and results.json.
    # Defaults to data/libero/runs/<YYYYMMDD_HHMMSS>_<task_suite_name> so each run
    # gets its own directory and nothing is ever clobbered.
    # Pass an explicit path to override (e.g. --args.video-out-path data/libero/my_run).
    video_out_path: str | None = None
    results_out_path: str | None = None  # Optional path to save aggregated results JSON

    # Optional JSON file that maps task_instruction -> logic_task_description.
    # Expected format matches libero_object_train7_logic_descriptions.json:
    #   {"tasks": [{"task_instruction": "...", "logic_task_description": "..."}, ...]}
    # When provided, any task whose natural-language instruction matches an entry will
    # have its prompt replaced with the corresponding logic_task_description.
    # Tasks with no entry in the file keep their default natural-language prompt.
    prompt_override_file: str | None = None

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")
    selected_tasks = libero_utils.resolve_task_filters(
        task_suite_name=args.task_suite_name,
        task_indices=args.task_indices,
        task_names=args.task_names,
        task_split_file=args.task_split_file,
        task_split=args.task_split,
    )
    selected_task_set = {task.casefold() for task in selected_tasks}
    selected_task_ids = [
        task_id
        for task_id in range(num_tasks_in_suite)
        if not selected_task_set or task_suite.get_task(task_id).language.casefold() in selected_task_set
    ]
    if not selected_task_ids:
        raise ValueError("No LIBERO evaluation tasks matched the requested task filter.")

    prompt_overrides: dict[str, str] = {}
    if args.prompt_override_file:
        with open(args.prompt_override_file) as f:
            override_data = json.load(f)
        prompt_overrides = {
            entry["task_instruction"]: entry["logic_task_description"]
            for entry in override_data["tasks"]
        }
        logging.info(f"Loaded {len(prompt_overrides)} prompt overrides from {args.prompt_override_file}")

    if args.video_out_path:
        video_out_path = pathlib.Path(args.video_out_path)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_tag = args.task_suite_name
        if args.prompt_override_file:
            run_tag += "_logic"
        video_out_path = pathlib.Path("data/libero/runs") / f"{timestamp}_{run_tag}"
    video_out_path.mkdir(parents=True, exist_ok=True)
    logging.info(f"Run output directory: {video_out_path}")

    results_out_path = pathlib.Path(args.results_out_path) if args.results_out_path else video_out_path / "results.json"
    results_out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    task_results = []
    for task_id in tqdm.tqdm(selected_task_ids):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_description = prompt_overrides.get(task_description, task_description)

        # Start episodes
        task_episodes, task_successes = 0, 0
        episode_results = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            done = False
            error = None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    error = str(e)
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = video_out_path / f"rollout_{task_segment}_{suffix}.mp4"
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
            episode_results.append(
                {
                    "episode_index": episode_idx,
                    "success": bool(done),
                    "steps_taken": t,
                    "video_path": str(video_path),
                    "error": error,
                }
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        task_results.append(
            {
                "task_id": task_id,
                "task_description": task_description,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": float(task_successes) / float(task_episodes),
                "episode_results": episode_results,
            }
        )

    total_success_rate = float(total_successes) / float(total_episodes)
    logging.info(f"Total success rate: {total_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_suite_name": args.task_suite_name,
        "task_split_file": args.task_split_file,
        "task_split": args.task_split,
        "selected_task_ids": selected_task_ids,
        "selected_tasks": selected_tasks,
        "num_trials_per_task": args.num_trials_per_task,
        "num_steps_wait": args.num_steps_wait,
        "replan_steps": args.replan_steps,
        "resize_size": args.resize_size,
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "video_out_path": str(video_out_path),
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "task_results": task_results,
    }
    results_out_path.write_text(json.dumps(results, indent=2) + "\n")
    logging.info(f"Saved results JSON to {results_out_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
