import json

from openpi.training import libero


def test_resolve_task_instructions_from_indices():
    tasks = libero.resolve_task_instructions("libero_object", task_indices=(0, 2, 4))

    assert tasks == (
        "pick up the alphabet soup and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the ketchup and place it in the basket",
    )


def test_resolve_task_instructions_accepts_ids_and_language():
    tasks = libero.resolve_task_instructions(
        "libero_object",
        task_names=(
            "pick_up_the_milk_and_place_it_in_the_basket",
            "pick up the orange juice and place it in the basket",
        ),
    )

    assert tasks == (
        "pick up the milk and place it in the basket",
        "pick up the orange juice and place it in the basket",
    )


def test_write_and_load_task_split(tmp_path):
    output_path = tmp_path / "split.json"
    split = libero.write_task_split(output_path, suite_name="libero_object", train_task_indices=(0, 1, 2))

    loaded = libero.load_task_split(output_path)

    assert loaded == split
    assert loaded.eval_task_indices == tuple(range(3, 10))
    assert json.loads(output_path.read_text())["suite_name"] == "libero_object"


def test_select_episode_indices():
    episodes = (
        {"episode_index": 0, "tasks": ["pick up the milk and place it in the basket"]},
        {"episode_index": 1, "tasks": ["pick up the orange juice and place it in the basket"]},
        {"episode_index": 2, "tasks": ["turn on the stove"]},
    )

    selected = libero.select_episode_indices(
        episodes,
        (
            "pick_up_the_orange_juice_and_place_it_in_the_basket",
            "turn on the stove",
        ),
    )

    assert selected == [1, 2]
