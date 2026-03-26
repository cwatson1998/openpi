from openpi.training import libero_logic


def test_extract_bddl_goal():
    bddl_text = """
    (define (problem test)
      (:goal
        (And (In alphabet_soup_1 basket_1_contain_region))
      )
    )
    """
    assert libero_logic.extract_bddl_goal(bddl_text) == "(And (In alphabet_soup_1 basket_1_contain_region))"


def test_infer_libero_suite_name_from_local_dataset_name():
    assert libero_logic.infer_libero_suite_name("local/libero_object_train7") == "libero_object"


def test_build_task_prompt_map_from_dataset_tasks(tmp_path):
    path = tmp_path / "logic.json"
    path.write_text(
        """
{
  "dataset_name": "local/test",
  "suite_name": "libero_object",
  "tasks": [
    {
      "task_id": "pick_up_the_ketchup_and_place_it_in_the_basket",
      "task_instruction": "pick up the ketchup and place it in the basket",
      "bddl_file": "/tmp/example.bddl",
      "goal_description": "(And (In ketchup_1 basket_1_contain_region))",
      "logic_task_description": "goal(in(ketchup_1,basket_1_contain_region))"
    }
  ]
}
""".strip()
        + "\n"
    )
    prompt_map = libero_logic.build_task_prompt_map_from_dataset_tasks(
        {0: "pick up the ketchup and place it in the basket"},
        path,
    )
    assert prompt_map == {0: "goal(in(ketchup_1,basket_1_contain_region))"}
