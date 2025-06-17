#!/bin/bash

# Evaluate all composite tasks from VLABench
# Composite tasks include: texas_holdem, texas_holdem_explore, set_dining_table, set_dining_left_hand, 
# set_dining_chopstick, cool_drink, take_out_cool_drink, heat_food, get_coffee, get_coffee_with_sugar, 
# get_coffee_with_milk, hammer_loose_nail, assemble_hammer, hammer_nail_and_hang_picture, make_juice, 
# simple_seesaw_use, complex_seesaw_use, set_study_table

python examples/vlabench/eval.py \
    --args.tasks="texas_holdem texas_holdem_explore set_dining_table set_dining_left_hand set_dining_chopstick cool_drink take_out_cool_drink heat_food get_coffee get_coffee_with_sugar get_coffee_with_milk hammer_loose_nail assemble_hammer hammer_nail_and_hang_picture make_juice simple_seesaw_use complex_seesaw_use set_study_table" \
    --args.episode_config_path="/remote-home1/sdzhang/project/VLABench/track_composite.json" \
    --args.save_dir="data/vlabench/pi0/track_composite_2_episode" \
    --args.n_episode=2 \
    --visulization