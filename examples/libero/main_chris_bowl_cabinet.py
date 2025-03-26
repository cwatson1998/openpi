# THis is super hacked to do a decomposition of one task.
import collections
import dataclasses
import logging
import math
import pathlib
from typing import Optional, List
from robosuite.wrappers import DataCollectionWrapper, Wrapper

LARGE_INT = 999999999999

#HACKY_BOWL_PROMPT = 'put the black bowl in the bottom drawer of the cabinet'
#HACKY_DRAWER_PROMPT = 'close the cabinet'

# This version uses the original prompt
#HACKY_BOWL_PROMPT = 'put the black bowl in the bottom drawer of the cabinet and close it'
#HACKY_DRAWER_PROMPT = 'put the black bowl in the bottom drawer of the cabinet and close it'

# nice prompt
#HACKY_BOWL_PROMPT = 'put the bowl in the drawer'
#HACKY_DRAWER_PROMPT = 'close the drawer'

HACKY_BOWL_PROMPT = 'put the black bowl in the bottom drawer of the cabinet'
HACKY_DRAWER_PROMPT = 'close the bottom drawer of the cabinet'

@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    task_list: Optional[List[str]] = None  # Explicitly typed list
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "data/libero/videos"
    seed: int = 7

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


""" class InfoWrapper(Wrapper):
    '''
    This does not work because robosuite does not support nested wrapping.
    In particular, getting attributes seems to not work.

    All this does is put things in info, which is useful because DataCollectionWrapper will save them.
    Not sure if it is ok robosuite usage to double wrap an environment.
    '''
    def __init__(self, env, info_obs_keys=None):
        super().__init__(env)
        if isinstance(info_obs_keys, str):
            info_obs_keys = [info_obs_keys]
        self.info_obs_keys = info_obs_keys
        

    def step(self, action):
        obs, rew, done, info = super().step(action)
        if self.info_obs_keys is not None:        
            for k in self.info_obs_keys:
                assert k not in info.keys()
                info[k] = obs[k]
        return obs, rew, done, info
 """



"""Scratch about put the black bowl in the bottom drawer of the cabinet and close it
    (:init
    (On akita_black_bowl_1 kitchen_table_akita_black_bowl_init_region)
    (On wine_bottle_1 kitchen_table_wine_bottle_init_region)
    (On white_cabinet_1 kitchen_table_white_cabinet_init_region)
    (On wine_rack_1 kitchen_table_wine_rack_init_region)
    (Open white_cabinet_1_bottom_region)
  )

  (:goal
    (And (Close white_cabinet_1_bottom_region) (In akita_black_bowl_1 white_cabinet_1_bottom_region))
  ) """


# Here I am going to define the functions that are actually useful for me and hardcode them in.
# In the future it would be good to get them from the bddl file or something.

# def get_eval_predicate_function(predicate_name, *args):
    

def get_eval_predicate_function(predicate_tuple):
    ''' 
    TODO: Rewrite this to be less cursed. Although, maybe it is impossible because
    Libero's OffscreenRenderEnv is not a Wrapper.
    For a binary predicate, predicate_tuple is:
        - name : str (e.g. 'on')
        - arg1 : str
        - arg2 : str'''
    def eval_predicate_function(env, predicate_tuple=predicate_tuple):
        try:
            return env._eval_predicate(predicate_tuple)
        except AttributeError:
            #print('debug: trying again')
            try:
                return env.env._eval_predicate(predicate_tuple)
            except AttributeError:
                return env.env.env._eval_predicate(predicate_tuple)
    return eval_predicate_function



class InfoDataCollectionWrapper(DataCollectionWrapper):
    """
        Really, I want to put a wrapper underneath a vanilla DataCollectionWrapper that puts
        relevant things in info, becaust DataCollectionWrapper will save everything in info.
        But robosuite's Wrapper class does not allow ``double wrapping'' so I have to do this.

        
        Annoyingly, I only save the stuff into the action_infos field. The trajectory will
        have one more state than action_info. The info fields correspond to the next_state.

        Args:
            info_obs_keys: list(str): Keys from the dict obs to get copied over into action_infos
            info_obs_function_dict: dict(str, observation -> ndarray)
            info_env_function_dict: dict(str, env -> ndarray)
    """
    def __init__(self, env, directory, extend_info=True, collect_freq=1, flush_freq=100, info_obs_keys=None, info_env_function_dict=None):#, info_obs_function_dict=None, info_env_function_dict=None):
        super().__init__(env, directory, collect_freq=collect_freq, flush_freq=flush_freq)
        if isinstance(info_obs_keys, str):
            info_obs_keys = [info_obs_keys]
        self.info_obs_keys = info_obs_keys
        #self.info_obs_function_dict=info_obs_function_dict
        self.info_env_function_dict=info_env_function_dict
        self.extend_info = extend_info
        if self.extend_info:
            # This is a horrible hack.
            # I need to refactor to include the functionality of both extending info and saving it.
            # I should probably just extend the info first and then save all the info.
            # Once i do that, the collect freq will not matter
            assert collect_freq==1
    
        # TODO: check that there are no overlapping keys, and that 'actions' does not appear.
        
        

    def step(self, action):
        """


        Extends vanilla step() function call to accommodate data collection

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (OrderedDict) observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        # This is important to skip DataCollectionWrapper's implementation
        ret = Wrapper.step(self, action)
        self.t += 1

        # on the first time step, make directories for logging
        if not self.has_interaction:
            self._on_first_interaction()

        # collect the current simulation state if necessary
        if self.t % self.collect_freq == 0:
            state = self.env.sim.get_state().flatten()
            self.states.append(state)
            
            info = {}
            info["actions"] = np.array(action)
            if self.info_obs_keys is not None:
                for k in self.info_obs_keys:
                    info[k] = ret[0][k]
            if self.info_env_function_dict is not None:
                for k, v in self.info_env_function_dict.items():
                    info[k] = v(self.env)
                    # TODO this is a quick hack
                    ret[-1][k] = v(self.env)
            #print(f'debug: {info}')
            self.action_infos.append(info)

        # check if the demonstration is successful
        try: 
            if self.env._check_success():
                self.successful = True
        except AttributeError:
            # Added by Chris because libero's OffscreenRenderEnv does not have _check_success,
            # because they don't extend RoboSuite's Wrapper class.
            if self.env.check_success():
                self.successful = True

        # flush collected data to disk if necessary
        if self.t % self.flush_freq == 0:
            self._flush()

        
                
        return ret





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
    # task_list: list[str] | None = None  # I think in python <3.9 the type annotation has to be task_list: Optional[List[str]] = None
    #task_id_list = None # Subset of tasks in the suite.
    task_id_list: Optional[List[int]] = None # Subset of tasks in the suite.
    # Be careful for libero_10, the task_order argument you pass to it will change this args meeting.
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    traj_out_path: Optional[str] = None # Path to save trajectories (using DataCollectionWrapper)
    traj_obs_info_key_list = None # Will pull these fields from the obs dict and put them in the action_info that gets saved.


    seed: int = 7  # Random Seed (for reproducibility)



def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)


    


    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

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
    task_successes_dict = dict()


    # Make a list of task_id, task pairs.
    if args.task_id_list is None:
        task_id_list = list(range(num_tasks_in_suite))
    else:
        task_id_list = args.task_id_list



    #for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    for task_id in task_id_list:
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_successes_dict[task_description] = 0
        task_successes_dict['bowl'] = 0

        # Wrap if saving
        if args.traj_out_path is not None:
            #env = InfoWrapper(env, info_obs_keys=['agentview_image'])
            info_obs_keys = None
            info_env_function_dict = {
                'In akita_black_bowl_1 white_cabinet_1_bottom_region': get_eval_predicate_function(('in', 'akita_black_bowl_1', 'white_cabinet_1_bottom_region'))
            }
            env = InfoDataCollectionWrapper(env, args.traj_out_path, flush_freq=LARGE_INT, info_obs_keys=info_obs_keys, info_env_function_dict=info_env_function_dict)

        # Start episodes
        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            hacky_finished_place = False
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

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
                            "prompt": HACKY_DRAWER_PROMPT if hacky_finished_place else HACKY_BOWL_PROMPT,
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
                    if info['In akita_black_bowl_1 white_cabinet_1_bottom_region']:
                        hacky_finished_place = True
                    #print(f'debug: info: {info}')
                    if done:
                        task_successes += 1
                        total_successes += 1
                        task_successes_dict[task_description] += 1
                        

                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            if hacky_finished_place:
                task_successes_dict['bowl'] += 1
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_{episode_idx}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Close env (important for flushing if saving trajectories)
    env.close()
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    print(HACKY_BOWL_PROMPT)
    print(HACKY_DRAWER_PROMPT)
    for k, v in task_successes_dict.items():
        print(f"{k}: {v}")


def _get_libero_env(task, resolution, seed, traj_out_path=None):
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
