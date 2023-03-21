import time
from copy import deepcopy
from typing import Dict, Any

import gym
import neurotorch as nt
import numpy as np
import torch
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import ActionTuple, BehaviorSpec, DimensionProperty
from neurotorch.callbacks.early_stopping import EarlyStoppingThreshold
from neurotorch.rl.utils import space_to_continuous_shape
from torchvision.transforms import Compose, Lambda

from PythonAcademy.neurotorch_fixes import RLAcademy, PPO
from PythonAcademy.src.utils import send_parameter_to_channel, threshold_image


class UnityWrapperToGymnasiumWrapper(gym.Wrapper):
	def __init__(self, env, observation_specs):
		super().__init__(env)
		self._env = env
		self.observation_specs = observation_specs
	
	def obs_list_to_dict(self, obs_list):
		obs_dict = {
			spec.name: obs
			for spec, obs in zip(self.observation_specs, obs_list)
		}
		return obs_dict
	
	def step(self, action):
		obs, reward, terminated, info = self._env.step(action)
		obs = self.obs_list_to_dict(obs)
		return obs, reward, terminated, False, info
	
	def reset(self):
		obs = self._env.reset()
		obs = self.obs_list_to_dict(obs)
		return obs, {}


def get_env_parameters(n_stack_input: int, **kwargs):
	default_params = dict(
		n_agents=1,
		camFollowTargetAgent=False,
		droneMinStartY=1.0,
		droneMaxStartY=2.5,
		observationStacks=n_stack_input,
		observationWidth=8,
		observationHeight=8,
		enableNeuromorphicCamera=True,
		enableCamera=False,
		usePositionAsInput=False,
		enableTorque=True,
		enableDisplacement=False,
		useRotationAsInput=False,
		useVelocityAsInput=False,
		useAngularVelocityAsInput=False,
		useDivergenceAsInput=True,
		divergenceAsOneHot=True,
		divergenceBins=20,
		divergenceBinSize=1.0,
		maxDivergencePoints=100,
		droneDrag=5.0,
		droneAngularDrag=7.0,
	)
	default_params.update(kwargs)
	return default_params


def setup_environment(
		*,
		build_path: str = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe",
		n_stack_input: int,
		params: Dict[str, Any] = None
):
	params_channel = EnvironmentParametersChannel()
	engine_config_channel = EngineConfigurationChannel()
	env = UnityEnvironment(
		file_name=build_path,
		seed=42,
		side_channels=[params_channel, engine_config_channel],
		no_graphics=False
	)
	engine_config_channel.set_configuration(
		EngineConfig(
			width=726, height=512, quality_level=1, time_scale=20.0, target_frame_rate=-1, capture_frame_rate=60
		)
	)
	env_params = get_env_parameters(n_stack_input)
	if params is not None:
		env_params.update(params)
	params_to_send = deepcopy(env_params)
	params_to_send["n_agents"] = 1
	sent_params = send_parameter_to_channel(params_channel, params_to_send)
	time.sleep(0.5)
	env.reset()
	time.sleep(0.5)
	return env, dict(params_channel=params_channel, engine_config_channel=engine_config_channel), env_params


def get_input_transforms(parameters: Dict[str, Any]):
	input_transform = []
	if np.isclose(float(parameters.get("enableNeuromorphicCamera", False)), 1.0):
		input_transform.append(
			Compose([
				Lambda(lambda a: nt.to_tensor(a, dtype=torch.float32)),
				threshold_image,
				Lambda(lambda t: torch.permute(t, (0, 3, 1, 2))),
				Lambda(lambda t: torch.flatten(t, start_dim=2))
			])
		)
	if np.isclose(float(parameters.get("useDivergenceAsInput", False)), 1.0):
		input_transform.append(
			Compose([
				Lambda(lambda a: nt.to_tensor(a, dtype=torch.float32)),
				Lambda(lambda t: torch.permute(t, (0, 3, 1, 2))),
				Lambda(lambda t: torch.flatten(t, start_dim=2))
			])
		)
	input_transform.append(
		Compose([
			Lambda(lambda a: nt.to_tensor(a, dtype=torch.float32)),
		])
	)
	return input_transform


def make_env(output_dict: dict, input_params: Dict[str, Any] = None):
	unity_env, channels, env_params = setup_environment(
		build_path="../MAVControlWithSNN/Builds/MAVControlWithSNN.exe",
		n_stack_input=10,
		params=input_params
	)
	env_id = list(unity_env.behavior_specs)[0].split("?")[0]
	env = UnityWrapperToGymnasiumWrapper(
		UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=True),
		unity_env.behavior_specs[list(unity_env.behavior_specs)[0]].observation_specs
	)
	output_dict["env"] = env
	output_dict["env_id"] = env_id
	output_dict["channels"] = channels
	output_dict["env_params"] = env_params
	output_dict["unity_env"] = unity_env
	return env


if __name__ == '__main__':
	unity_env, channels, env_params = setup_environment(
		build_path="../MAVControlWithSNN/Builds/MAVControlWithSNN.exe",
		n_stack_input=10,
		params=dict(
			enableTorque=False,
			droneMaxStartY=2.5,
		)
	)
	env_id = list(unity_env.behavior_specs)[0].split("?")[0]
	env = UnityWrapperToGymnasiumWrapper(
		UnityToGymWrapper(unity_env, uint8_visual=False, flatten_branched=False, allow_multiple_obs=True),
		unity_env.behavior_specs[list(unity_env.behavior_specs)[0]].observation_specs
	)
	# trial_parameters = dict(
	# 	n_agents=4,
	# 	enableTorque=False,
	# 	droneMaxStartY=2.5,
	# )
	# worker_infos = [dict() for _ in range(trial_parameters["n_agents"])]
	# envs = gym.vector.SyncVectorEnv([
	# 	lambda: make_env(worker_infos[i], deepcopy(trial_parameters))
	# 	for i in range(len(worker_infos))
	# ])
	# continuous_obs_shape = space_to_continuous_shape(getattr(env, "single_observation_space", env.observation_space))
	# continuous_action_shape = space_to_continuous_shape(getattr(env, "single_action_space", env.action_space))
	continuous_obs_shape = [obs.shape for obs in env.observation_space.spaces]
	continuous_action_shape = env.action_space.shape
	
	n_hidden_units = 128
	n_critic_hidden_units = 128
	last_k_rewards = 100
	n_iterations = int(1e4)
	n_epochs = 30
	n_new_trajectories = 100 * env_params["n_agents"]
	
	input_sizes = {
		obs.name: np.prod([
			d
			for d, d_type in zip(obs.shape, obs.dimension_property)
			if d_type == DimensionProperty.TRANSLATIONAL_EQUIVARIANCE
		], dtype=int)
		for obs in unity_env.behavior_specs[list(unity_env.behavior_specs)[0]].observation_specs
	}
	
	policy = nt.SequentialRNN(
		input_transform=get_input_transforms(env_params),
		layers=[
			{
				key: nt.SpyLIFLayer(
					int(obs_shape), n_hidden_units, use_recurrent_connection=False
				)
				for key, obs_shape in input_sizes.items()
			},
			nt.SpyLILayer(int(n_hidden_units * len(input_sizes)), continuous_action_shape[0]),
		],
		output_transform=[nt.transforms.ReduceMean(dim=1)],
	).build()
	
	ppo_la = PPO(
		critic_criterion=torch.nn.SmoothL1Loss(),
	)
	
	agent = nt.rl.Agent(
		env=env,
		behavior_name=env_id,
		policy=policy,
		critic=nt.Sequential(
			input_transform=[
				Compose(
					[
						Lambda(lambda a: nt.to_tensor(a, dtype=torch.float32)),
						Lambda(lambda t: t[..., -1]),
						Lambda(lambda t: torch.flatten(t, start_dim=1))
					]
				)
				for _ in range(len(input_sizes))
			],
			layers=[
				{
					key: torch.nn.Sequential(
						torch.nn.Linear(int(obs_shape), n_critic_hidden_units),
						torch.nn.Dropout(0.1),
						torch.nn.PReLU(),
					)
					for key, obs_shape in input_sizes.items()
				},
				torch.nn.Linear(int(n_critic_hidden_units*len(input_sizes)), n_critic_hidden_units),
				torch.nn.Dropout(0.1),
				torch.nn.PReLU(),
				torch.nn.Linear(n_critic_hidden_units, 1),
			]
		).build(),
		checkpoint_folder=f"data/tr_data/ckps_{env_id}_snn-policy",
		continuous_action_variances_decay=1 - (0.5 / n_iterations),
	)
	
	checkpoint_manager = nt.CheckpointManager(
		checkpoint_folder=agent.checkpoint_folder,
		save_freq=int(0.1 * n_iterations),
		metric=nt.rl.RLAcademy.CUM_REWARDS_METRIC_KEY,
		minimise_metric=False,
		save_best_only=True,
	)
	
	early_stopping = EarlyStoppingThreshold(
		metric=f"mean_last_{last_k_rewards}_rewards",
		threshold=0.99,
		minimize_metric=False,
	)
	# es_timer = nt.callbacks.early_stopping.EarlyStoppingOnTimeLimit(delta_seconds=60 * 60 * 1)  # 1 h
	academy = RLAcademy(
		agent=agent,
		callbacks=[checkpoint_manager, ppo_la, early_stopping],
	)
	print(f"Academy:\n{academy}")
	
	history = academy.train(
		env,
		n_iterations=n_iterations,
		n_epochs=n_epochs,
		n_batches=-1,
		n_new_trajectories=n_new_trajectories,
		batch_size=4096,
		# buffer_size=int(1e4),
		# clear_buffer=False,
		# use_priority_buffer=True,
		randomize_buffer=True,
		load_checkpoint_mode=nt.LoadCheckpointMode.LAST_ITR,
		force_overwrite=False,
		verbose=True,
		render=False,
		last_k_rewards=last_k_rewards,
	)
	
	agent.load_checkpoint(
		checkpoints_meta_path=checkpoint_manager.checkpoints_meta_path,
		load_checkpoint_mode=nt.LoadCheckpointMode.BEST_ITR
	)
	env.reset()
	agent.eval()
	gen_trajectories_out = academy.generate_trajectories(
		n_trajectories=10, epsilon=0.0, verbose=True, env=env, render=True, re_trajectories=True,
	)
	best_trajectory_idx = np.argmax([t.cumulative_reward for t in gen_trajectories_out.trajectories])
	trajectory_renderer = nt.rl.utils.TrajectoryRenderer(
		trajectory=gen_trajectories_out.trajectories[best_trajectory_idx], env=env
	)
	
	cumulative_rewards = gen_trajectories_out.cumulative_rewards
	print(f"Buffer: {gen_trajectories_out.buffer}")
	print(f"Cumulative rewards: {np.nanmean(cumulative_rewards):.3f} +/- {np.nanstd(cumulative_rewards):.3f}")
	best_cum_reward_fmt = f"{cumulative_rewards[best_trajectory_idx]:.3f}"
	print(f"Best trajectory: {best_trajectory_idx}, cumulative reward: {best_cum_reward_fmt}")
	
	if not getattr(env, "closed", False):
		env.close()
	
	fig, ax, anim = trajectory_renderer.render(
		filename=(
			f"{agent.checkpoint_folder}/figures/trajectory_{best_trajectory_idx}-"
			f"CR{best_cum_reward_fmt.replace('.', '_')}"
		),
		file_extension="gif",
		show=False,
	)
