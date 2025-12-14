# MIT License

# Copyright (c) 2025 ReinFlow Authors

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




"""
Parent eval agent class with state input, for openai-gym environment
"""
import os
import numpy as np
import torch
import hydra
import logging
import random
from tqdm import tqdm as tqdm
log = logging.getLogger(__name__)
from env.gym_utils import make_async
from omegaconf import OmegaConf
import torch.nn as nn
import os
import cv2
from agent.eval.visualize.utils import read_eval_statistics
from util.dirs import REINFLOW_DIR 
class EvalAgent:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.device = cfg.device
        self.base_policy_path = cfg.base_policy_path
        if not self.base_policy_path:
            raise ValueError("base_policy_path must be set in the config file!")
        self.eval_log_dir = cfg.get('eval_log_dir', None)
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        ############ could be overload #############
        self.record_video = False
        self.frame_width = 640  # Default, can be overridden
        self.frame_height = 480
        self.all_video_paths=[] # a list of video paths for each denoising step.
        self.record_env_index = 0
        self.render_onscreen = False
        self.denoising_steps = None
        self.denoising_steps_trained = None
        self.plot_scale = cfg.get("plot_scale", "semilogx")  # Default to semilogx, can be "standard" or "semilogx"
        self.plot_scale_options=['standard, semilogx']
        # if self.plot_scale not in self.plot_scale_options:
        #     raise ValueError(f"plot scale must be one of {self.plot_scale_options}, but received {self.plot_scale}!")
        ############################################
        
        # Make vectorized env
        self.env_name: str = cfg.env.name
        env_type = cfg.env.get("env_type", None)
        self.venv = make_async(
            cfg.env.name,
            env_type=env_type,
            num_envs=cfg.env.n_envs,
            asynchronous=True,
            max_episode_steps=cfg.env.max_episode_steps,
            wrappers=cfg.env.get("wrappers", None),
            robomimic_env_cfg_path=cfg.get("robomimic_env_cfg_path", None),
            shape_meta=cfg.get("shape_meta", None),
            use_image_obs=cfg.env.get("use_image_obs", False),
            render=cfg.env.get("render", False),
            render_offscreen=cfg.env.get("save_video", False),
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            **cfg.env.specific if "specific" in cfg.env else {},
        )
        if not env_type == "furniture":
            self.venv.seed(
                [self.seed + i for i in range(cfg.env.n_envs)]
            )
        self.n_envs = cfg.env.n_envs
        self.n_cond_step = cfg.cond_steps
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim
        self.act_steps = cfg.act_steps
        self.horizon_steps = cfg.horizon_steps
        self.max_episode_steps = cfg.env.max_episode_steps
        self.reset_at_iteration = cfg.env.get("reset_at_iteration", True)
        self.furniture_sparse_reward = (
            cfg.env.specific.get("sparse_reward", False)
            if "specific" in cfg.env
            else False
        )
        
        # Build model and load checkpoint
        self.model = hydra.utils.instantiate(cfg.model)
        
        # Eval params
        self.n_steps = cfg.n_steps
        self.best_reward_threshold_for_success = (
            len(self.venv.pairs_to_assemble)
            if env_type == "furniture"
            else cfg.env.best_reward_threshold_for_success
        )

        # Logging, rendering
        self.logdir = cfg.logdir
        self.render_dir = os.path.join(self.logdir, "render")
        self.result_path = os.path.join(self.logdir, "result.npz")
        os.makedirs(self.render_dir, exist_ok=True)
        self.n_render = cfg.render_num
        self.render_video = cfg.env.get("save_video", False)
        assert self.n_render <= self.n_envs, "n_render must be <= n_envs"
        assert not (
            self.n_render <= 0 and self.render_video
        ), "Need to set n_render > 0 if saving video"
        
        # Rollout saving for fiper
        self.save_rollouts = cfg.get("save_rollouts", False)
        self.save_rollouts_for_steps = cfg.get("save_rollouts_for_steps", [])
        self.save_rollouts_dir = cfg.get("save_rollouts_dir", "rollouts")
        self.calibration_episodes_limit = cfg.get("calibration_episodes_limit", 50)  # Number of successful episodes for calibration
        self.action_pred_batch_size = cfg.get("action_pred_batch_size", 32)  # Number of action predictions per state for action_pred
        # OOD initial state configuration
        self.ood_initial_state_ratio = cfg.get("ood_initial_state_ratio", 0.5)  # Fraction of environments/episodes to use OOD initial states
        if self.save_rollouts:
            # Create directory structure: rollouts/calibration/, rollouts/test/, rollouts/videos/calibration/, rollouts/videos/test/
            self.save_rollouts_calibration_dir = os.path.join(self.save_rollouts_dir, "calibration")
            self.save_rollouts_test_dir = os.path.join(self.save_rollouts_dir, "test")
            self.save_videos_calibration_dir = os.path.join(self.save_rollouts_dir, "videos", "calibration")
            self.save_videos_test_dir = os.path.join(self.save_rollouts_dir, "videos", "test")
            os.makedirs(self.save_rollouts_calibration_dir, exist_ok=True)
            os.makedirs(self.save_rollouts_test_dir, exist_ok=True)
            os.makedirs(self.save_videos_calibration_dir, exist_ok=True)
            os.makedirs(self.save_videos_test_dir, exist_ok=True)
        self.episode_counter = 0
        self.calibration_episodes_count = 0  # Track number of successful episodes saved to calibration
        # Track initial state types per environment: env_ind -> 'id' or 'ood'
        # Each environment maintains a fixed type throughout the run
        self.env_initial_state_types = {}
        
    
    def load_model_for_eval(self):
        data = torch.load(self.base_policy_path, weights_only=True, map_location=self.device)
        self.model: nn.Module        
        print(f"loading model...")
        if self.load_ema:
            if 'ema' in data.keys():
                if any('network' in key for key in data["ema"].keys()):
                    self.model.load_state_dict(data["ema"], strict=False)
                else:
                    actor_policy_state_dict = {key.replace('actor_ft.policy.', 'network.'): value 
                                        for key, value in data["ema"].items() 
                                        if key.startswith('actor_ft.policy.')}
                    if actor_policy_state_dict == {}:
                        raise ValueError(f"No parameter starting with actor_ft.policy in ={data['ema'].keys()}")
                    self.model.load_state_dict(actor_policy_state_dict, strict=False)
            else:
                raise ValueError(f"You set self.load_ema={self.load_ema}, but your state dictionary does not contain key: ema. It only contains keys: {data.keys()}")
            log.info(f"Loaded EMA model dict from {self.base_policy_path}")
        else:
            if 'policy' in data.keys():
                self.model.load_state_dict(data["policy"], strict=True)
            if 'ema' in data.keys():
                if any('network' in key for key in data["ema"].keys()):
                    self.model.load_state_dict(data["ema"], strict=True)
            elif 'model' in data.keys():
                actor_policy_state_dict = {} 
                for key, value in data["model"].items():
                    if key.startswith('actor_ft.mlp_logvar') or key.startswith('actor_ft.logvar') or key.startswith('actor_ft.explore_noise_net.'):
                        continue
                    if key.startswith('actor_ft.policy.'):
                        actor_policy_state_dict[key.replace('actor_ft.policy.', 'network.')] = value
                    elif key.startswith('actor_ft.'):
                        actor_policy_state_dict[key.replace('actor_ft.', 'network.')] = value
                if actor_policy_state_dict == {}:
                    raise ValueError(f"No parameter starting with actor_ft.policy or actor_ft. in ={data['model'].keys()}")
                self.model.load_state_dict(actor_policy_state_dict, strict=True)
            else:
                raise ValueError(f"Your state dictionary is not correct, it does not contain keys: policy or model. It only contains keys: {data.keys()}")
            log.info(f"Loaded model dict from {self.base_policy_path}")

    
    def reset_env_all(self, verbose=False, options_venv=None, **kwargs):
        if options_venv is None:
            options_venv = [
                {k: v for k, v in kwargs.items()} for _ in range(self.n_envs)
            ]
        obs_venv = self.venv.reset_arg(options_list=options_venv)
        if isinstance(obs_venv, list):
            obs_venv = {
                key: np.stack([obs_venv[i][key] for i in range(self.n_envs)])
                for key in obs_venv[0].keys()
            }
        if verbose:
            for index in range(self.n_envs):
                logging.info(
                    f"<-- Reset environment {index} with options {options_venv[index]}"
                )
        return obs_venv

    def reset_env(self, env_ind, verbose=False):
        task = {}
        obs = self.venv.reset_one_arg(env_ind=env_ind, options=task)
        if verbose:
            logging.info(f"<-- Reset environment {env_ind} with task {task}")
        return obs
    
    
    def run(self):
        if self.render_onscreen and self.n_envs > 1:
            raise ValueError(f"Cannot render on screen with more than one parallel envs. self.render_onscreen={self.render_onscreen}, cfg.env.n_envs={self.n_envs}")
        if self.eval_log_dir is None:
            self.eval_log_dir = f'visualize/{self.model.__class__.__name__}/{self.env_name}/{self.current_time()}/'
        os.makedirs(self.eval_log_dir, exist_ok=True)
        cfg_path = os.path.join(self.eval_log_dir, "cfg.yaml")
        with open(cfg_path, 'w') as f:
            OmegaConf.save(self.cfg, f)
        print(f"Configuration saved to {cfg_path}")
        
        options_venv = [{} for _ in range(self.n_envs)]
        if self.render_video:
            for env_ind in range(self.n_render):
                options_venv[env_ind]["video_path"] = os.path.join(
                    self.render_dir, f"eval_trial-{env_ind}.mp4"
                )
        
        self.load_model_for_eval()
        denoising_steps_set = self.denoising_steps
        
        # Lists to store the results
        num_denoising_steps_list = []
        avg_single_step_freq_list = []
        avg_single_step_freq_std_list = []
        avg_single_step_duration_list = []
        avg_single_step_duration_std_list = []
        avg_traj_length_list = []
        avg_traj_length_std_list = []  # Added
        avg_episode_reward_list = []
        avg_episode_reward_std_list = []
        avg_best_reward_list = []
        avg_best_reward_std_list = []
        success_rate_list = []
        success_rate_std_list = []  # Added
        num_episodes_finished_list = []
        
        for num_denoising_steps in denoising_steps_set:
            self.venv.reset()
            result = self.single_run(num_denoising_steps, options_venv)
            
            num_denoising_steps, avg_single_step_freq, avg_single_step_freq_std, \
                avg_single_step_duration, avg_single_step_duration_std, \
                avg_traj_length, avg_traj_length_std, \
                avg_episode_reward, avg_episode_reward_std, \
                avg_best_reward, avg_best_reward_std, \
                num_episodes_finished, success_rate, success_rate_std = result
            
            num_denoising_steps_list.append(num_denoising_steps)
            avg_single_step_freq_list.append(avg_single_step_freq)
            avg_single_step_freq_std_list.append(avg_single_step_freq_std)
            avg_single_step_duration_list.append(avg_single_step_duration)
            avg_single_step_duration_std_list.append(avg_single_step_duration_std)
            avg_traj_length_list.append(avg_traj_length)
            avg_traj_length_std_list.append(avg_traj_length_std)  # Added
            avg_episode_reward_list.append(avg_episode_reward)
            avg_episode_reward_std_list.append(avg_episode_reward_std)
            avg_best_reward_list.append(avg_best_reward)
            avg_best_reward_std_list.append(avg_best_reward_std)
            success_rate_list.append(success_rate)
            success_rate_std_list.append(success_rate_std)  # Added
            num_episodes_finished_list.append(num_episodes_finished)
        
        # Save evaluation statistics as an npz
        dtype = [
            ('num_denoising_steps', int),
            ('avg_single_step_freq', float),
            ('avg_single_step_freq_std', float),
            ('avg_single_step_duration', float),
            ('avg_single_step_duration_std', float),
            ('avg_traj_length', float),
            ('avg_traj_length_std', float),  # Added
            ('avg_episode_reward', float),
            ('avg_episode_reward_std', float),
            ('avg_best_reward', float),
            ('avg_best_reward_std', float),
            ('success_rate', float),
            ('success_rate_std', float),  # Added
            ('num_episodes_finished', int)
        ]

        data = np.zeros(len(num_denoising_steps_list), dtype=dtype)
        data['num_denoising_steps'] = num_denoising_steps_list
        data['avg_single_step_freq'] = avg_single_step_freq_list
        data['avg_single_step_freq_std'] = avg_single_step_freq_std_list
        data['avg_single_step_duration'] = avg_single_step_duration_list
        data['avg_single_step_duration_std'] = avg_single_step_duration_std_list
        data['avg_traj_length'] = avg_traj_length_list
        data['avg_traj_length_std'] = avg_traj_length_std_list  # Added
        data['avg_episode_reward'] = avg_episode_reward_list
        data['avg_episode_reward_std'] = avg_episode_reward_std_list
        data['avg_best_reward'] = avg_best_reward_list
        data['avg_best_reward_std'] = avg_best_reward_std_list
        data['success_rate'] = success_rate_list
        data['success_rate_std'] = success_rate_std_list  # Added
        data['num_episodes_finished'] = num_episodes_finished_list

        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')
        np.savez(eval_statistics_path, data=data)
        
        statistics = read_eval_statistics(npz_file_path=eval_statistics_path)
        self.plot_eval_statistics(statistics, self.eval_log_dir)


    def create_video_recorder(self, num_denoising_steps:int):
        self.video_writer = None
        if self.record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_path = os.path.join(self.eval_log_dir, f'{self.model.__class__.__name__}_{self.env_name}_step{num_denoising_steps}.mp4')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 20.0, (self.frame_width, self.frame_height))
            self.video_title = f"{self.model.__class__.__name__}, {num_denoising_steps} steps"
        
    def single_run(self, num_denoising_steps, options_venv):
        
        self.create_video_recorder(num_denoising_steps)
        
        # Reset initial state types tracking for this run
        self.env_initial_state_types = {}
        
        # Check if we should save rollouts for this denoising step
        should_save_rollouts = (self.save_rollouts and 
                               num_denoising_steps in self.save_rollouts_for_steps)
        
        self.model.eval()
        firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
        
        # Prepare options with initial state types (ID or OOD)
        # Determine which environments should use OOD initial states
        if should_save_rollouts:
            # Determine OOD environments based on ratio
            n_ood_envs = max(1, int(self.n_envs * self.ood_initial_state_ratio))
            ood_env_indices = np.random.choice(self.n_envs, size=n_ood_envs, replace=False).tolist()
            
            # Prepare options_venv with initial_state_type and OOD states
            prepared_options_venv = []
            for env_ind in range(self.n_envs):
                env_options = options_venv[env_ind].copy() if env_ind < len(options_venv) else {}
                
                if env_ind in ood_env_indices:
                    env_options['initial_state_type'] = 'ood'
                else:
                    env_options['initial_state_type'] = 'id'
                
                prepared_options_venv.append(env_options)
        else:
            prepared_options_venv = options_venv
        
        prev_obs_venv = self.reset_env_all(options_venv=prepared_options_venv)
        firsts_trajs[0] = 1
        
        # Track initial state types for each environment (fixed throughout the run)
        if should_save_rollouts:
            for env_ind in range(self.n_envs):
                initial_state_type = prepared_options_venv[env_ind].get('initial_state_type', 'id')
                self.env_initial_state_types[env_ind] = initial_state_type
        
        reward_trajs = np.zeros((self.n_steps, self.n_envs))
        single_step_duration_list = np.zeros(self.n_steps)
        
        # Data collection for rollouts
        if should_save_rollouts:
            rollout_data = {
                'obs': [],  # List of observations for each step
                'actions': [],  # List of executed actions
                'action_preds': [],  # List of full action predictions
                'obs_embeddings': [],  # List of observation embeddings
                'rewards': [],  # List of rewards
                'frames': []  # List of rendered frames for each step per environment
            }
        
        # Log message can remain as is
        log.info(f"Evaluating {self.model.__class__.__name__} model in {self.env_name} environment with {num_denoising_steps} step(s).")
        
        for step in tqdm(range(self.n_steps), dynamic_ncols=True, desc=f'{num_denoising_steps} step(s):'):
            with torch.no_grad():
                # Generic observation handling
                if hasattr(self, 'obs_dims'):  # For image-based agents
                    cond = {
                        key: torch.from_numpy(prev_obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                else:  # For state-based agents
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    }
                
                # Generate action predictions
                # If saving rollouts, generate multiple action predictions per state for action_pred
                # Otherwise, generate single sample for execution
                action_preds_batch = None
                obs_embeddings = None
                
                if should_save_rollouts:
                    # Generate action_pred_batch_size predictions for each environment
                    # Expand condition to (n_envs * action_pred_batch_size, ...) by repeating each env's state
                    if hasattr(self, 'obs_dims'):  # Image-based
                        cond_expanded = {
                            key: cond[key].repeat_interleave(self.action_pred_batch_size, dim=0)
                            for key in cond.keys()
                        }
                    else:  # State-based
                        cond_expanded = {
                            "state": cond["state"].repeat_interleave(self.action_pred_batch_size, dim=0)
                        }
                    
                    # Generate batch of predictions
                    samples_batch, single_step_duration = self.infer(cond_expanded, num_denoising_steps)
                    action_preds_all = samples_batch.trajectories.cpu().numpy()  # (n_envs * action_pred_batch_size, horizon_steps, action_dim)
                    
                    # Reshape to (n_envs, action_pred_batch_size, horizon_steps, action_dim)
                    action_preds_batch = action_preds_all.reshape(
                        self.n_envs, self.action_pred_batch_size, self.horizon_steps, self.action_dim
                    )
                    
                    # Use the first prediction from the batch for execution
                    output_venv = action_preds_batch[:, 0, :, :]  # (n_envs, horizon_steps, action_dim) - take first sample
                    
                    # Extract embeddings from the first sample
                    _, _, obs_embeddings = self.infer(cond, num_denoising_steps, extract_embeddings=True)
                else:
                    # Generate single sample for execution
                    samples, single_step_duration = self.infer(cond, num_denoising_steps)
                    output_venv = samples.trajectories.cpu().numpy()  # (n_envs, horizon_steps, action_dim)
                
                single_step_duration_list[step] = single_step_duration
                
                # Collect rollout data if needed
                if should_save_rollouts:
                    # Store observations (convert to numpy and handle format)
                    obs_dict = {k: prev_obs_venv[k].copy() for k in prev_obs_venv.keys()}
                    rollout_data['obs'].append(obs_dict)
                    # Store action_preds_batch: (n_envs, action_pred_batch_size, horizon_steps, action_dim)
                    rollout_data['action_preds'].append(action_preds_batch.copy())
                    if obs_embeddings is not None:
                        rollout_data['obs_embeddings'].append(obs_embeddings.copy())
                    
            # Execute the first action from the output (which is action_pred[0] when saving rollouts)
            action_venv = output_venv[:, : self.act_steps]
            
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                self.venv.step(action_venv)
            )
            
            # Collect action and reward data
            if should_save_rollouts:
                rollout_data['actions'].append(action_venv.copy())  # (n_envs, act_steps, action_dim)
                rollout_data['rewards'].append(reward_venv.copy())
            
            # Render and collect frames for video saving
            if should_save_rollouts:
                # Render frames for all environments
                frames = None
                try:
                    if 'kitchen' in self.env_name.lower():
                        # Kitchen environments need special handling
                        frames = None
                    else:
                        # Render for all environments
                        frame_tuple = self.venv.render(mode='rgb_array', height=self.frame_height, width=self.frame_width)
                        if frame_tuple is not None:
                            if isinstance(frame_tuple, (tuple, list)):
                                # Multiple frames, one per environment
                                frames = np.array(frame_tuple)  # (n_envs, H, W, C)
                            elif isinstance(frame_tuple, np.ndarray):
                                # Could be (n_envs, H, W, C) or (H, W, C)
                                if frame_tuple.ndim == 4:
                                    frames = frame_tuple  # (n_envs, H, W, C)
                                elif frame_tuple.ndim == 3:
                                    # Single frame - duplicate for all environments
                                    frames = np.stack([frame_tuple] * self.n_envs, axis=0)
                    rollout_data['frames'].append(frames)
                except Exception as e:
                    log.debug(f"Failed to render frame at step {step}: {e}")
                    rollout_data['frames'].append(None)
            
            if self.render_onscreen:
                self.venv.render(mode='human')
            if self.record_video:
                if 'kitchen' in self.env_name.lower(): # Kitchen
                    raise ValueError(f"Cannot record video for kitchen environments with the current setup. self.env_name={self.env_name}") # For kitchen environments, we render with the sim.render method, as D4RL kitchen does not support the standard render method.
                else: # gym or robomimic or d3il
                    frame_tuple = self.venv.render(mode='rgb_array', height=self.frame_height, width=self.frame_width)
                if self.video_writer is not None:
                    frame = frame_tuple[self.record_env_index]
                    # print(f"frame_tuple={len(frame_tuple)}, frame={frame.shape}, frame={frame}")
                    if frame is None or frame == []:
                        raise ValueError(f"frame is {frame} (empty), check your environment rendering settings.")
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # add title to indicate the model type and the number of denoising steps. 
                    cv2.putText(frame, self.video_title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    self.video_writer.write(frame)
            
            reward_trajs[step] = reward_venv
            done_venv = terminated_venv | truncated_venv
            firsts_trajs[step + 1] = done_venv
            
            # Note: When environments reset after being done, they maintain their fixed
            # initial state type (ID or OOD) that was set at initialization.
            # The wrapper remembers this type and uses it automatically.
            
            prev_obs_venv = obs_venv
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.all_video_paths.append(self.video_path)
            print(f"Video saved to {self.video_path}")

        episodes_start_end = []
        for env_ind in range(self.n_envs):
            env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
            for i in range(len(env_steps) - 1):
                start = env_steps[i]
                end = env_steps[i + 1]
                if end - start > 1:
                    episodes_start_end.append((env_ind, start, end - 1))
        
        # Save rollouts if enabled
        if should_save_rollouts and len(episodes_start_end) > 0:
            self._save_rollouts(rollout_data, episodes_start_end, reward_trajs, num_denoising_steps)
        
        if len(episodes_start_end) > 0:
            reward_trajs_split = [
                reward_trajs[start : end + 1, env_ind]
                for env_ind, start, end in episodes_start_end
            ]
            num_episodes_finished = len(reward_trajs_split)
            episode_reward = np.array(
                [np.sum(reward_traj) for reward_traj in reward_trajs_split]
            )
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.array(
                    [np.max(reward_traj) / self.act_steps for reward_traj in reward_trajs_split]
                )
            avg_episode_reward = np.mean(episode_reward)
            avg_episode_reward_std = np.std(episode_reward)
            avg_best_reward = np.mean(episode_best_reward)
            avg_best_reward_std = np.std(episode_best_reward)
            success_rate = np.mean(
                episode_best_reward >= self.best_reward_threshold_for_success
            )
            success_rate_std = np.std(episode_best_reward >= self.best_reward_threshold_for_success)
        else:
            episode_reward = np.array([])
            num_episodes_finished = 0
            avg_episode_reward = 0
            avg_episode_reward_std=0
            avg_best_reward = 0
            avg_best_reward_std=0
            success_rate = 0
            success_rate_std = 0  # Added
            log.info("[WARNING] No episode completed within the iteration!")
        
        episode_lengths = np.array([end - start + 1 for _, start, end in episodes_start_end]) * self.act_steps
        avg_traj_length = np.mean(episode_lengths) if len(episode_lengths) > 0 else 0
        avg_traj_length_std = np.std(episode_lengths) if len(episode_lengths) > 0 else 0  # Added
        
        avg_single_step_duration = single_step_duration_list.mean()
        avg_single_step_duration_std = single_step_duration_list.std()
        single_step_frequency_list = 1 / single_step_duration_list
        avg_single_step_freq = single_step_frequency_list.mean()
        avg_single_step_freq_std = single_step_frequency_list.std()
        
        BOLDSTART = '\033[1m'
        BOLDEND = '\033[0m'
        log.info(
            f"""
            #############################################################
            {BOLDSTART}Evaluation{BOLDEND}
            Model:                    {self.model.__class__.__name__:>30}
            Environment:              {self.env_name + ' x ' + str(self.n_envs):>30}
            denoising steps:          {num_denoising_steps:>30}
            
            success_rate:             {success_rate*100:>8.3f} % ± {success_rate_std*100:>8.3f} %
            avg_episode_reward:       {avg_episode_reward:>8.1f} ± {avg_episode_reward_std:>2.1f}
            
            
            avg_single_step_freq:     {avg_single_step_freq:>3.1f} ± {avg_single_step_freq_std:>3.1f} HZ
            
            avg_traj_length:          {avg_traj_length:>3.1f} ± {avg_traj_length_std:>3.1f} steps
            avg_best_reward:          {avg_best_reward:>8.1f} ± {avg_best_reward_std:>2.1f}
            num_episode:              {num_episodes_finished:>4d}
            #############################################################
            """
        )
        
        return num_denoising_steps, \
            avg_single_step_freq, avg_single_step_freq_std, \
            avg_single_step_duration, avg_single_step_duration_std, \
            avg_traj_length, avg_traj_length_std, \
            avg_episode_reward, avg_episode_reward_std, \
            avg_best_reward, avg_best_reward_std, \
            num_episodes_finished, success_rate, success_rate_std
    
    def _save_rollouts(self, rollout_data, episodes_start_end, reward_trajs, num_denoising_steps):
        """Save rollouts for each completed episode in fiper-compatible format."""
        import pickle
        import time as time_module
        
        for env_ind, start_step, end_step in episodes_start_end:
            # Calculate episode reward to determine success
            episode_reward_traj = reward_trajs[start_step:end_step+1, env_ind]
            episode_reward = np.sum(episode_reward_traj)
            if self.furniture_sparse_reward:
                episode_best_reward = episode_reward
            else:
                episode_best_reward = np.max(episode_reward_traj) / self.act_steps
            is_successful = episode_best_reward >= self.best_reward_threshold_for_success
            
            # Determine if this should go to calibration or test
            # First 50 successful episodes go to calibration, rest to test
            is_calibration = is_successful and (self.calibration_episodes_count < self.calibration_episodes_limit)
            if is_calibration:
                self.calibration_episodes_count += 1
            
            # Extract episode data
            episode_rollout = []
            num_episode_steps = end_step - start_step + 1
            
            for step_idx in range(start_step, end_step + 1):
                # Get observation data for this environment
                obs_dict = rollout_data['obs'][step_idx]
                
                # Extract RGB image for this environment
                rgb_img = None
                if 'rgb' in obs_dict:
                    rgb = obs_dict['rgb']
                    # rgb should be (n_envs, C, H, W) or (n_envs, T, C, H, W) if history
                    if rgb.ndim == 5:  # (n_envs, T, C, H, W) - take last timestep
                        rgb = rgb[:, -1]  # (n_envs, C, H, W)
                    if rgb.ndim == 4:  # (n_envs, C, H, W)
                        rgb_img = rgb[env_ind]  # (C, H, W)
                    elif rgb.ndim == 3:  # (C, H, W) - single image
                        rgb_img = rgb
                    
                    if rgb_img is not None:
                        # Convert (C, H, W) to (H, W, C) for fiper format
                        if rgb_img.shape[0] == 3 or rgb_img.shape[0] == 1:
                            rgb_img = np.transpose(rgb_img, (1, 2, 0))
                        # If grayscale, convert to RGB
                        if rgb_img.ndim == 2:
                            rgb_img = np.stack([rgb_img]*3, axis=-1)
                        elif rgb_img.ndim == 3 and rgb_img.shape[2] == 1:
                            rgb_img = np.repeat(rgb_img, 3, axis=-1)
                
                # Extract agent position (proprioceptive state)
                agent_pos = None
                if 'state' in obs_dict:
                    state = obs_dict['state']
                    # state should be (n_envs, T, obs_dim) or (n_envs, obs_dim)
                    if state.ndim == 3:  # (n_envs, T, obs_dim) - take last timestep
                        state = state[:, -1]  # (n_envs, obs_dim)
                    if state.ndim == 2:  # (n_envs, obs_dim)
                        agent_pos = state[env_ind]  # (obs_dim,)
                    elif state.ndim == 1:  # (obs_dim,)
                        agent_pos = state
                
                if agent_pos is None:
                    # Fallback: use zeros
                    agent_pos = np.zeros(self.obs_dim)
                
                # Get executed action - store as (act_steps, action_dim)
                action = rollout_data['actions'][step_idx][env_ind]  # (act_steps, action_dim)
                
                # Get action predictions - batch of predictions for this environment
                # action_preds has shape (n_envs, action_pred_batch_size, horizon_steps, action_dim)
                # Extract predictions for this specific environment
                action_pred = rollout_data['action_preds'][step_idx][env_ind]  # (action_pred_batch_size, horizon_steps, action_dim)
                
                # Get observation embedding
                obs_embedding = None
                if len(rollout_data['obs_embeddings']) > step_idx:
                    obs_emb = rollout_data['obs_embeddings'][step_idx]
                    if obs_emb is not None:
                        if obs_emb.ndim == 2:  # (n_envs, embedding_dim)
                            obs_embedding = obs_emb[env_ind]  # (embedding_dim,)
                        elif obs_emb.ndim == 1:  # (embedding_dim,)
                            obs_embedding = obs_emb
                
                # Create rollout step entry
                step_entry = {
                    'rgb': rgb_img,
                    'action': action,  # (act_steps, action_dim) - executed actions
                    'action_pred': action_pred,  # (n_envs, horizon_steps, action_dim) - full batch predictions
                    'agent_pos': agent_pos,
                    'obs_embedding': obs_embedding,
                    'state_embedding': None,  # Not used in current setup
                    'timestamp': step_idx * 0.1,  # Approximate timestamp
                    'step': step_idx - start_step
                }
                episode_rollout.append(step_entry)
            
            # Create metadata
            # Determine action mappings based on action_dim
            action_mappings = self._get_action_mappings()
            
            # Determine rollout type and directory based on calibration/test split
            if is_calibration:
                rollout_type = 'calibration'
                rollout_subtype = 'ca'  # Calibration
                save_dir = self.save_rollouts_calibration_dir
                video_save_dir = self.save_videos_calibration_dir
            else:
                rollout_type = 'test'
                # Get initial state type for this environment (ID or OOD)
                # Each environment maintains a fixed type throughout the run
                initial_state_type = self.env_initial_state_types.get(env_ind, 'id')
                rollout_subtype = initial_state_type  # 'id' or 'ood' based on initial state distribution
                save_dir = self.save_rollouts_test_dir
                video_save_dir = self.save_videos_test_dir
            
            metadata = {
                'metadata': True,
                'task': self.env_name,
                'episode': self.episode_counter,
                'num_robots': 1,
                'rollout_type': rollout_type,
                'rollout_subtype': rollout_subtype,
                'action_prediction_horizon': self.horizon_steps,
                'action_execution_horizon': self.act_steps,
                'action_batch_size': self.action_pred_batch_size,  # Batch size for action predictions
                'has_encoder_feat': True,  # We have obs_embeddings
                'has_state_feat': False,  # We don't have separate state embeddings
                'action_mappings': action_mappings,
                'successful': is_successful,
                'num_steps': num_episode_steps,
                'denoising_steps': num_denoising_steps  # Add denoising steps used
            }
            
            # Save episode
            suffix = 's' if is_successful else 'f'
            filename = f'episode_{suffix}_{self.episode_counter:04d}.pkl'
            filepath = os.path.join(save_dir, filename)
            
            rollout_dict = {
                'metadata': metadata,
                'rollout': episode_rollout
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(rollout_dict, f)
            
            # Save video for this episode
            self._save_episode_video(rollout_data, env_ind, start_step, end_step, is_successful, video_save_dir)
            
            self.episode_counter += 1
            log.info(f"Saved rollout to {filepath} (type={rollout_type}, success={is_successful}, steps={num_episode_steps}, calibration_count={self.calibration_episodes_count})")
    
    def _save_episode_video(self, rollout_data, env_ind, start_step, end_step, is_successful, video_save_dir):
        """Save video for a specific episode."""
        # Extract frames for this environment and episode
        episode_frames = []
        for step_idx in range(start_step, end_step + 1):
            frame = None
            
            # Try to get frame from rendered frames first
            if step_idx < len(rollout_data['frames']) and rollout_data['frames'][step_idx] is not None:
                frames = rollout_data['frames'][step_idx]
                # frames can be (n_envs, H, W, C) or (H, W, C)
                if frames.ndim == 4:  # (n_envs, H, W, C)
                    frame = frames[env_ind]  # (H, W, C)
                elif frames.ndim == 3:  # (H, W, C) - single frame
                    frame = frames
            
            # Fallback: use RGB observation if available
            if frame is None and step_idx < len(rollout_data['obs']):
                obs_dict = rollout_data['obs'][step_idx]
                if 'rgb' in obs_dict:
                    rgb = obs_dict['rgb']
                    # rgb can be (n_envs, C, H, W) or (n_envs, T, C, H, W)
                    if rgb.ndim == 5:  # (n_envs, T, C, H, W)
                        rgb = rgb[:, -1]  # Take last timestep
                    if rgb.ndim == 4:  # (n_envs, C, H, W)
                        rgb_img = rgb[env_ind]  # (C, H, W)
                        # Convert (C, H, W) to (H, W, C)
                        if rgb_img.shape[0] == 3:
                            frame = np.transpose(rgb_img, (1, 2, 0))
                        else:
                            frame = rgb_img
                    elif rgb.ndim == 3:  # (C, H, W)
                        if rgb.shape[0] == 3:
                            frame = np.transpose(rgb, (1, 2, 0))
                        else:
                            frame = rgb
            
            if frame is not None:
                # Ensure frame is in correct format (H, W, C) with values 0-255
                if frame.dtype != np.uint8:
                    # Normalize if needed (assuming 0-1 range or 0-255 range)
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = np.clip(frame, 0, 255).astype(np.uint8)
                
                # Ensure 3 channels
                if frame.ndim == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                elif frame.ndim == 3 and frame.shape[2] == 1:
                    frame = np.repeat(frame, 3, axis=-1)
                
                # Convert RGB to BGR for cv2
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                episode_frames.append(frame_bgr)
        
        if len(episode_frames) == 0:
            log.warning(f"No frames collected for episode {self.episode_counter}, skipping video save")
            return
        
        # Determine video dimensions
        h, w = episode_frames[0].shape[:2]
        
        # Create video filename
        suffix = 's' if is_successful else 'f'
        video_filename = f'episode_{suffix}_{self.episode_counter:04d}.mp4'
        video_filepath = os.path.join(video_save_dir, video_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0  # Frames per second
        video_writer = cv2.VideoWriter(video_filepath, fourcc, fps, (w, h))
        
        if not video_writer.isOpened():
            log.error(f"Failed to open video writer for {video_filepath}")
            return
        
        # Write frames
        for frame in episode_frames:
            # Resize if needed to match video dimensions
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h))
            video_writer.write(frame)
        
        video_writer.release()
        log.info(f"Saved video to {video_filepath} ({len(episode_frames)} frames)")
    
    def _get_action_mappings(self):
        """Get action mappings based on action dimension. Default mapping for 7-dim actions."""
        # Default mapping for 7-dim actions (pos, quat/rot, gripper)
        if self.action_dim == 7:
            return {
                'pos': [0, 1, 2],
                'rpy': [3, 4, 5],
                'gripper': [6],
                'vel': None,
                'angular_velocity': None,
                'joint_pos': None,
                'joint_vel': None,
                'drpy': None
            }
        else:
            # Generic mapping - all actions
            return {
                'pos': list(range(min(3, self.action_dim))),
                'rpy': list(range(3, min(6, self.action_dim))) if self.action_dim > 3 else None,
                'gripper': [self.action_dim - 1] if self.action_dim > 0 else None,
                'vel': None,
                'angular_velocity': None,
                'joint_pos': None,
                'joint_vel': None,
                'drpy': None
            }
    
    def current_time(self):
        from datetime import datetime
        now = datetime.now()
        formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
        return formatted_time
    
    def plot_eval_statistics(self, eval_statistics, log_dir: str):
        num_denoising_steps_list, \
            avg_single_step_freq_list, avg_single_step_freq_std_list, \
            avg_single_step_duration_list, avg_single_step_duration_std_list, \
            avg_traj_length_list, avg_traj_length_std_list, \
            avg_episode_reward_list, avg_episode_reward_std_list, \
            avg_best_reward_list, avg_best_reward_std_list, \
            success_rate_list, success_rate_std_list, \
            num_episodes_finished_list = eval_statistics
        
        import matplotlib.pyplot as plt
        import os
        
        plt.figure(figsize=(12, 8))
        
        # Define plotting function based on self.plot_scale
        plot_func = plt.semilogx if self.plot_scale == "semilogx" else plt.plot

         # Plot success rate
        plt.subplot(2, 3, 1)
        plot_func(num_denoising_steps_list, success_rate_list, marker='o', label='Success Rate', color='y')
        plt.fill_between(num_denoising_steps_list,
                        [sr - std for sr, std in zip(success_rate_list, success_rate_std_list)],
                        [sr + std for sr, std in zip(success_rate_list, success_rate_std_list)],
                        color='y', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Success Rate')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.legend()


        # Plot average episode reward
        plt.subplot(2, 3, 2)
        plot_func(num_denoising_steps_list, avg_episode_reward_list, marker='o', label='Avg Episode Reward', color='b')
        plt.fill_between(num_denoising_steps_list,
                        [avg_episode - std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                        [avg_episode + std for avg_episode, std in zip(avg_episode_reward_list, avg_episode_reward_std_list)],
                        color='b', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Episode Reward')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Episode Reward')
        plt.grid(True)
        plt.legend()

        # Plot average trajectory length
        plt.subplot(2, 3, 5)
        plot_func(num_denoising_steps_list, avg_traj_length_list, marker='o', label='Avg Trajectory Length', color='r')
        plt.fill_between(num_denoising_steps_list,
                        [avg_traj - std for avg_traj, std in zip(avg_traj_length_list, avg_traj_length_std_list)],
                        [avg_traj + std for avg_traj, std in zip(avg_traj_length_list, avg_traj_length_std_list)],
                        color='r', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Trajectory Length')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Trajectory Length')
        plt.grid(True)
        plt.legend()

        # Plot inference duration
        plt.subplot(2, 3, 3)
        plot_func(num_denoising_steps_list, avg_single_step_duration_list, marker='o', label='Time', color='purple')
        plt.fill_between(num_denoising_steps_list,
                        [duration - std for duration, std in zip(avg_single_step_duration_list, avg_single_step_duration_std_list)],
                        [duration + std for duration, std in zip(avg_single_step_duration_list, avg_single_step_duration_std_list)],
                        color='purple', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Duration')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Time (s)')
        plt.grid(True)
        plt.legend()
        
        
        # Plot average best reward
        plt.subplot(2, 3, 4)
        plot_func(num_denoising_steps_list, avg_best_reward_list, marker='o', label='Avg Best Reward', color='g')
        plt.fill_between(num_denoising_steps_list,
                        [avg_best - std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                        [avg_best + std for avg_best, std in zip(avg_best_reward_list, avg_best_reward_std_list)],
                        color='g', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Average Best Reward')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Average Best Reward')
        plt.grid(True)
        plt.legend()

       
        # Plot inference frequency
        plt.subplot(2, 3, 6)
        plot_func(num_denoising_steps_list, avg_single_step_freq_list, marker='o', label='Frequency', color='brown')
        plt.fill_between(num_denoising_steps_list,
                        [freq - std for freq, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                        [freq + std for freq, std in zip(avg_single_step_freq_list, avg_single_step_freq_std_list)],
                        color='brown', alpha=0.2, label='Std Dev')
        if self.denoising_steps_trained:
            plt.axvline(x=self.denoising_steps_trained, color='black', linestyle='--', label='Training Steps')
        plt.title('Inference Frequency')
        plt.xlabel('Number of Denoising Steps')
        plt.ylabel('Inference Frequency (Hz)')
        plt.grid(True)
        plt.legend()

        plt.suptitle(f"{self.model.__class__.__name__}, {self.env_name}\nsteps = {', '.join(map(str, num_denoising_steps_list))}", fontsize=25)
        plt.tight_layout()
        
        eval_statistics_path = os.path.join(self.eval_log_dir, 'eval_statistics.npz')


        fig_path = os.path.join(REINFLOW_DIR, log_dir, f'denoise_step.png')
        plt.savefig(fig_path)
        print(f"Finished evaluating {self.model.__class__.__name__} in environment {self.env_name}")
        print(f"Base_policy_path: {os.path.join(REINFLOW_DIR,self.base_policy_path)}")
        print(f"Figure saved to {fig_path}")
        print(f"Evaluation statistics saved to  {eval_statistics_path}")
        if self.record_video:
            print(f"Video(s) saved to {self.all_video_paths}")   
        plt.close()