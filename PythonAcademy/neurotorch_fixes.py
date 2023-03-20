from collections import defaultdict
from copy import deepcopy
from typing import Tuple, Optional, Any, List, Dict, Iterator, Iterable

import gym
import numpy as np
import torch
from neurotorch import to_numpy, to_tensor, ToDevice
from neurotorch.rl.rl_academy import GenTrajectoriesOutput
from neurotorch.rl.utils import env_batch_reset as nt_env_batch_reset, batch_numpy_actions, env_batch_render
from neurotorch.rl import RLAcademy as nt_RLAcademy, Trajectory, Experience
from neurotorch.rl import ReplayBuffer as nt_ReplayBuffer
from neurotorch.rl.rl_academy import AgentsHistoryMaps as nt_AgentsHistoryMaps
from neurotorch.rl.buffers import BatchExperience as nt_BatchExperience
from neurotorch.rl import PPO as nt_PPO
from tqdm import tqdm


def batch_dict_of_items(x: Any) -> Any:
	if isinstance(x, dict):
		return {k: batch_dict_of_items(v) for k, v in x.items()}
	else:
		return np.array([x])
	
	
def get_item_from_batch(x: Any, i: int) -> Any:
	if isinstance(x, dict):
		return {k: get_item_from_batch(v, i) for k, v in x.items()}
	else:
		return x[i]


def env_batch_step(
		env: gym.Env,
		actions: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""
	Step the environment in batch mode.

	:param env: The environment.
	:type env: gym.Env
	:param actions: The actions to take.
	:type actions: Any

	:return: The batch of observations, rewards, dones, truncated and infos.
	:rtype: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
	"""
	# actions_as_numpy = to_numpy(actions).reshape(-1).tolist()  # TODO: to numpy without changing the dtype
	if isinstance(actions, dict) and len(actions) == 1:
		actions = list(actions.values())[0]
	actions_as_numpy = actions
	if isinstance(env, gym.vector.VectorEnv):
		observations, rewards, dones, truncateds, info = env.step(actions_as_numpy)
		infos = [info for _ in range(env.num_envs)]
	else:
		actions_as_single = actions_as_numpy[0] if actions_as_numpy.ndim > 0 else actions_as_numpy
		observation, reward, done, truncated, info = env.step(actions_as_single)
		observations = batch_dict_of_items(observation)
		rewards = np.array([reward])
		dones = np.array([done])
		truncateds = np.array([truncated])
		infos = np.array([info])
	return observations, rewards, dones, truncateds, infos


def env_batch_reset(env: gym.Env) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Reset the environment in batch mode.

	:param env: The environment.
	:type env: gym.Env

	:return: The batch of observations.
	:rtype: np.ndarray
	"""
	if isinstance(env, gym.vector.VectorEnv):
		observations, infos = env.reset()
	else:
		observation, info = env.reset()
		observations = batch_dict_of_items(observation)
		infos = np.array([info])
	return observations, infos


class BatchExperience(nt_BatchExperience):
	def __init__(
			self,
			batch: List[Experience],
			device: torch.device = torch.device("cpu"),
	):
		"""
		An object that contains a batch of experiences as tensors.

		:param batch: A list of Experience objects.
		:param device: The device to use for the tensors.
		"""
		batch = deepcopy(batch)
		self._batch = batch
		self._device = device
		self._to = ToDevice(device=device)

		self.obs: List[torch.Tensor] = self._make_obs_batch(batch)
		self.rewards: torch.Tensor = self._make_rewards_batch(batch)
		self.terminals: torch.Tensor = self._make_terminals_batch(batch)
		self.actions = self._make_actions_batch(batch)
		self.next_obs: List[torch.Tensor] = self._make_next_obs_batch(batch)
		self.others: List[dict] = [ex.others for ex in batch]
	
	def _make_obs_batch(self, batch: List[Experience]) -> List[torch.Tensor]:
		as_dict = isinstance(batch[0].obs, dict)
		# [{k: x.shape for k, x in exp.obs.items()} for exp in batch]
		if as_dict:
			obs = {
				key: torch.stack([to_tensor(ex.obs[key]) for ex in batch])
				for key in batch[0].obs
			}
			return self._to(obs)
		return self._to(torch.stack([to_tensor(ex.obs) for ex in batch]))
	
	def _make_next_obs_batch(self, batch: List[Experience]) -> List[torch.Tensor]:
		as_dict = isinstance(batch[0].next_obs, dict)
		if as_dict:
			obs = {
				key: torch.stack([to_tensor(ex.next_obs[key]) for ex in batch])
				for key in batch[0].next_obs
			}
			return self._to(obs)
		return self._to(torch.stack([to_tensor(ex.next_obs) for ex in batch]))
	
	def _make_actions_batch(self, batch: List[Experience]) -> torch.Tensor:
		as_dict = isinstance(batch[0].action, dict)
		if as_dict:
			action = {
				key: torch.stack([to_tensor(ex.action[key]) for ex in batch])
				for key in batch[0].action
			}
			return self._to(action)
		return self._to(torch.stack([to_tensor(ex.action) for ex in batch]))


class ReplayBuffer(nt_ReplayBuffer):
	def extend(self, iterable: Iterable[Experience]):
		_ = [self.store(e) for e in iterable]
		return self
	
	def store(self, element: Experience):
		"""
		Stores an element. If the replay buffer is already full, deletes the oldest
		element to make space.
		"""
		if len(self.data) >= self.__capacity:
			if self.use_priority:
				self.data.pop(np.argmin([np.abs(getattr(e, self.priority_key, 0.0)) for e in self.data]))
			else:
				self.data.pop(0)
		self.data.append(deepcopy(element))
		if self._counter_is_started:
			self._counter += 1


class AgentsHistoryMaps(nt_AgentsHistoryMaps):
	def update_trajectories_(
			self,
			*,
			observations,
			actions,
			next_observations,
			rewards,
			terminals,
			truncated=None,
			infos=None,
			others=None,
	) -> List[Trajectory]:
		"""
		Updates the trajectories of the agents and returns the trajectories of the agents that have been terminated.

		:param observations: The observations
		:param actions: The actions
		:param next_observations: The next observations
		:param rewards: The rewards
		:param terminals: The terminals
		:param truncated: The truncated
		:param infos: The infos
		:param others: The others

		:return: The terminated trajectories.
		"""
		actions = deepcopy(to_numpy(actions))
		observations, next_observations = deepcopy(to_numpy(observations)), deepcopy(to_numpy(next_observations))
		rewards, terminals = deepcopy(to_numpy(rewards)), deepcopy(to_numpy(terminals))
		if others is None:
			others = [None] * len(observations)
		self.min_rewards = min(self.min_rewards, np.min(rewards))
		self.max_rewards = max(self.max_rewards, np.max(rewards))
		if self.normalize_rewards:
			rewards = rewards / (self.max_abs_rewards + 1e-8)
		
		finished_trajectories = []
		for i in range(len(terminals)):
			if self.trajectories[i].terminated:
				continue
			if terminals[i]:
				self.trajectories[i].append_and_propagate(
					Experience(
						obs=get_item_from_batch(observations, i),
						reward=rewards[i],
						terminal=terminals[i],
						action=get_item_from_batch(actions, i),
						next_obs=get_item_from_batch(next_observations, i),
						others=get_item_from_batch(others, i),
					)
				)
				self.cumulative_rewards[i].append(self.trajectories[i].cumulative_reward)
				self.terminal_rewards[i] = self.trajectories[i].terminal_reward
				finished_trajectory = self.trajectories.pop(i)
				finished_trajectories.append(finished_trajectory)
				# self.buffer.extend(finished_trajectory)
				self._terminal_counter += 1
				self._experience_counter += 1
			else:
				self.trajectories[i].append(
					Experience(
						obs=get_item_from_batch(observations, i),
						reward=rewards[i],
						terminal=terminals[i],
						action=get_item_from_batch(actions, i),
						next_obs=get_item_from_batch(next_observations, i),
						others=get_item_from_batch(others, i),
					)
				)
				self._experience_counter += 1
		return finished_trajectories
	
	def propagate_and_get_all(self) -> List[Trajectory]:
		"""
		Propagate all the trajectories and return all the trajectories.

		:return: All the trajectories
		:rtype: List[Trajectory]
		"""
		trajectories = []
		for i in range(len(self.trajectories)):
			if not self.trajectories[i].propagated:
				self.trajectories[i].propagate()
			if self.trajectories[i].terminated:
				self.cumulative_rewards[i].append(self.trajectories[i].cumulative_reward)
				trajectory = self.trajectories.pop(i)
				self._terminal_counter += 1
			else:
				trajectory = self.trajectories[i]
			trajectories.append(trajectory)
			# self.buffer.extend(trajectory)
		return trajectories
	
	def propagate_all(self) -> List[Trajectory]:
		"""
		Propagate all the trajectories and return the finished ones.

		:return: All the trajectories.
		:rtype: List[Trajectory]
		"""
		trajectories = []
		for i in range(len(self.trajectories)):
			if not self.trajectories[i].propagated:
				self.trajectories[i].propagate()
			if self.trajectories[i].terminated:
				self.cumulative_rewards[i].append(self.trajectories[i].cumulative_reward)
				trajectory = self.trajectories.pop(i)
				trajectories.append(trajectory)
				self._terminal_counter += 1
			else:
				trajectory = self.trajectories[i]
			# self.buffer.extend(trajectory)
		return trajectories
	
	def clear(self) -> List[Trajectory]:
		trajectories = self.propagate_and_get_all()
		self.trajectories.clear()
		self.cumulative_rewards.clear()
		self._terminal_counter = 0
		self._experience_counter = 0
		return trajectories


class RLAcademy(nt_RLAcademy):
	def generate_trajectories(
			self,
			*,
			n_trajectories: Optional[int] = None,
			n_experiences: Optional[int] = None,
			buffer: Optional[ReplayBuffer] = None,
			epsilon: float = 0.0,
			p_bar_position: int = 0,
			verbose: Optional[bool] = None,
			**kwargs
	) -> GenTrajectoriesOutput:
		"""
		Generate trajectories using the current policy. If the policy of the agent is in evaluation mode, the
		actions will be chosen with the argmax method. If the policy is in training mode and a random number is
		generated that is less than epsilon, a random action will be chosen. Otherwise, the action will be chosen
		by a sample considering the policy output.

		:param n_trajectories: Number of trajectories to generate. If not specified, the number of trajectories
			will be calculated based on the number of experiences.
		:type n_trajectories: int
		:param n_experiences: Number of experiences to generate. If not specified, the number of experiences
			will be calculated based on the number of trajectories.
		:type n_experiences: int
		:param buffer: The buffer to store the experiences.
		:type buffer: ReplayBuffer
		:param epsilon: The probability of choosing a random action.
		:type epsilon: float
		:param p_bar_position: The position of the progress bar.
		:type p_bar_position: int
		:param verbose: Whether to show the progress bar.
		:type verbose: bool
		:param kwargs: Additional arguments.

		:keyword gym.Env env: The environment to generate the trajectories. Will update the "env" of the current_state.
		:keyword observation: The initial observation. If not specified, the observation will be get from the
			the objects of the current_state attribute and if not available, the environment will be reset.
		:keyword info: The initial info. If not specified, the info will be get from the objects of the
			current_state attribute and if not available, the environment will be reset.

		:return: The buffer with the generated experiences, the cumulative rewards and the mean of terminal rewards.
		"""
		if n_trajectories is None:
			n_trajectories = self.kwargs["n_new_trajectories"]
		if n_experiences is None:
			n_experiences = self.kwargs["n_new_experiences"]
		if buffer is None:
			buffer = self.current_training_state.objects.get(
				"buffer",
				ReplayBuffer(
					self.kwargs["buffer_size"],
					use_priority=self.kwargs["use_priority_buffer"],
					priority_key=self.kwargs["buffer_priority_key"],
				)
			)
			self.update_objects_state_(buffer=buffer)
		if self.kwargs["clear_buffer"]:
			buffer.clear()
		if verbose is None:
			verbose = self.verbose
		if "env" in kwargs:
			if self.state.objects.get("env", None) != kwargs["env"]:
				self.reset_agents_history_maps_meta()
			self.update_objects_state_(env=kwargs["env"])
		render = kwargs.get("render", self.kwargs.get("render", False))
		rendering = [None for _ in range((self.env.num_envs if hasattr(self.env, "num_envs") else 1))]
		re_trajectories = kwargs.get("re_trajectories", False)
		re_trajectories_list = []
		agents_history_maps = AgentsHistoryMaps(
			# buffer,
			normalize_rewards=self.kwargs["normalize_rewards"], **self._agents_history_maps_meta
		)
		# terminal_rewards: List[float] = []
		p_bar = tqdm(
			total=n_experiences if n_trajectories is None else n_trajectories,
			disable=not verbose, desc="Generating Trajectories", position=p_bar_position,
			unit="trajectory" if n_trajectories is not None else "experience",
		)
		observations = kwargs.get("observations", self.current_training_state.objects.get("observations", None))
		info = kwargs.get("info", self.current_training_state.objects.get("info", None))
		if observations is None or info is None:
			observations, info = env_batch_reset(self.env)
		while not self._update_gen_trajectories_break_flag(agents_history_maps, n_trajectories, n_experiences):
			if render:
				rendering = env_batch_render(self.env)
			if not self.agent.training:
				actions = self.agent.get_actions(observations, env=self.env, re_format="argmax", as_numpy=True)
			elif np.random.random() < epsilon:
				actions = self.agent.get_random_actions(env=self.env, re_format="argmax", as_numpy=True)
			else:
				actions = self.agent.get_actions(observations, env=self.env, re_format="sample", as_numpy=True)
			actions = batch_numpy_actions(actions, self.env)
			next_observations, rewards, dones, truncated, info = env_batch_step(self.env, actions)
			terminals = np.logical_or(dones, truncated)
			finished_trajectories = agents_history_maps.update_trajectories_(
				observations=observations,
				actions=actions,
				next_observations=next_observations,
				rewards=rewards,
				terminals=terminals,
				others=[
					{"render": rendering_item, "info": info_item, "truncated": truncated_item}
					for rendering_item, info_item, truncated_item in zip(rendering, info, truncated)
				],
			)
			# terminal_rewards = list(agents_history_maps.terminal_rewards.values())
			if all(terminals):
				finished_trajectories.extend(agents_history_maps.propagate_all())
				next_observations, info = env_batch_reset(self.env)
			self._update_gen_trajectories_finished_trajectories(finished_trajectories, buffer)
			if re_trajectories:
				re_trajectories_list.extend(finished_trajectories)
			if n_trajectories is None:
				p_bar.update(min(len(terminals), max(0, n_experiences - len(terminals))))
			else:
				p_bar.update(min(sum(terminals), max(0, n_trajectories - sum(terminals))))
			p_bar.set_postfix(
				cumulative_reward=f"{agents_history_maps.mean_cumulative_rewards:.3f}",
				# terminal_rewards=f"{np.nanmean(terminal_rewards) if terminal_rewards else 0.0:.3f}",
			)
			observations = next_observations
		self._update_gen_trajectories_finished_trajectories(agents_history_maps.propagate_and_get_all(), buffer)
		self._update_agents_history_maps_meta(agents_history_maps)
		self.update_objects_state_(observations=observations, info=info, buffer=buffer)
		self.update_itr_metrics_state_(
			**{
				self.CUM_REWARDS_METRIC_KEY: agents_history_maps.mean_cumulative_rewards,
				# self.TERMINAL_REWARDS_METRIC_KEY: np.mean(terminal_rewards),
			}
		)
		p_bar.close()
		return GenTrajectoriesOutput(
			buffer=buffer,
			cumulative_rewards=agents_history_maps.cumulative_rewards_as_array,
			agents_history_maps=agents_history_maps,
			trajectories=re_trajectories_list,
		)
	
	def _update_gen_trajectories_finished_trajectories(
			self,
			finished_trajectories: List[Trajectory],
			buffer: Optional[ReplayBuffer] = None,
	):
		if buffer is None:
			buffer = self.state.objects.get("buffer", None)
		for trajectory in finished_trajectories:
			if not trajectory.is_empty():
				trajectory_others_list = self.callbacks.on_trajectory_end(self, trajectory)
				if trajectory_others_list is not None:
					trajectory.update_others(trajectory_others_list)
				if buffer is not None:
					buffer.extend(trajectory)
		return buffer
	
	
class PPO(nt_PPO):
	def _batch_obs(self, batch: List[Experience]):
		as_dict = isinstance(batch[0].obs, dict)
		if as_dict:
			obs_batched = batch[0].obs
			for key in obs_batched:
				obs_batched[key] = torch.stack([to_tensor(ex.obs[key]) for ex in batch]).to(self.policy.device)
		else:
			obs_batched = torch.stack([to_tensor(ex.obs) for ex in batch]).to(self.policy.device)
		return obs_batched
	
	def _compute_values(self, trajectory):
		obs_as_tensor = BatchExperience(trajectory).obs
		values = self.agent.get_values(obs_as_tensor, as_numpy=True, re_as_dict=False).reshape(-1)
		return values
	

nt_PPO = PPO
nt_RLAcademy = RLAcademy
nt_BatchExperience = BatchExperience
nt_ReplayBuffer = ReplayBuffer
