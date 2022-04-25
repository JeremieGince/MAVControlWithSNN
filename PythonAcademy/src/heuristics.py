import os
from typing import List, Tuple

import numpy as np
from mlagents_envs.base_env import ActionTuple, BaseEnv

from tqdm.auto import tqdm

from PythonAcademy.src.buffers import ReplayBuffer
from PythonAcademy.src.snn_agent import AgentsHistoryMaps, SNNAgent


class Heuristic:
	def __init__(
			self,
			env: BaseEnv,
			name: str = "Heuristic",
			n_trajectories: int = int(2**13),
			trajectories_folder: str = None,
			generate_trajectories: bool = True,
			force_generate_trajectories: bool = False,
			verbose: bool = True,
	):
		self.env = env
		self.spec = env.behavior_specs[list(env.behavior_specs)[0]]
		self.name = name
		self.n_obs = n_trajectories
		self.verbose = verbose
		if trajectories_folder is None:
			trajectories_folder = self.name + "_trajectories"
		self.trajectories_folder = trajectories_folder
		assert '.' not in self.trajectories_folder, "Folder name should not contain '.'"
		self.generate_trajectories_flag = generate_trajectories
		self.force_generate_trajectories = force_generate_trajectories
		self._buffer = None

	@property
	def buffer(self) -> ReplayBuffer:
		if self._buffer is None:
			self.load_buffer()
		return self._buffer

	def load_buffer(self):
		if self.force_generate_trajectories:
			self._buffer = ReplayBuffer(self.n_obs)
			self.generate_trajectories(self.env, self.verbose)
		else:
			self._buffer = self.load_trajectories()
			if self.generate_trajectories_flag and len(self._buffer) < self.n_obs:
				if self._buffer.capacity < self.n_obs:
					self._buffer.increase_capacity(self.n_obs - self._buffer.capacity)
				self.generate_trajectories(self.env, self.verbose)
		self.save_trajectories()

	def discard_buffer(self):
		del self._buffer
		self._buffer = None

	def forward(self, inputs, **kwargs):
		raise NotImplementedError

	def get_actions(self, inputs, **kwargs) -> ActionTuple:
		return ActionTuple(continuous=self.forward(inputs), discrete=None)

	def __call__(self, *args, **kwargs) -> ActionTuple:
		return self.get_actions(*args, **kwargs)

	def save_trajectories(self):
		os.makedirs(self.trajectories_folder, exist_ok=True)
		self._buffer.save(f"{self.trajectories_folder}/buffer.pkl")

	def load_trajectories(self):
		if not os.path.exists(f"{self.trajectories_folder}/buffer.pkl"):
			return ReplayBuffer(self.n_obs)
		buffer = ReplayBuffer.load(f"{self.trajectories_folder}/buffer.pkl")
		self._buffer = buffer
		return buffer

	def generate_trajectories(
			self,
			env: BaseEnv,
			verbose: bool = True,
	) -> Tuple[ReplayBuffer, List[float]]:
		env.reset()
		behavior_name = list(env.behavior_specs)[0]
		agent_maps = AgentsHistoryMaps()
		cumulative_rewards: List[float] = []
		p_bar = tqdm(total=self.n_obs, disable=not verbose, desc=f"Generating Observations - {self.name}")
		last_buffer_counter = 0
		while len(self._buffer) < self.n_obs:  # While not enough data in the buffer
			decision_steps, terminal_steps = env.get_steps(behavior_name)
			cumulative_rewards.extend(SNNAgent.exec_terminal_steps_(agent_maps, self._buffer, terminal_steps))
			if len(decision_steps) > 0:
				SNNAgent.exec_decisions_steps_(agent_maps, decision_steps)
				actions = self.get_actions(decision_steps.obs)
				actions_list = SNNAgent.unbatch_actions(actions)
				for agent_index, agent_id in enumerate(decision_steps.agent_id):
					agent_maps.last_action[agent_id] = actions_list[agent_index]
				env.set_actions(behavior_name, actions)
			p_bar.set_postfix(cumulative_reward=f"{np.mean(cumulative_rewards) if cumulative_rewards else 0.0:.3f}")
			p_bar.update(len(self._buffer) - last_buffer_counter)
			last_buffer_counter = len(self._buffer)
			env.step()
		p_bar.close()
		return self._buffer, cumulative_rewards


class FgHeuristic(Heuristic):
	def __init__(
			self,
			env: BaseEnv,
			name: str = "Fg_heuristic",
			mass: float = 1.0,
			noise: float = 0.1,
			engine_force: float = 20.0,
			n_trajectories: int = 4096,
			trajectories_folder: str = None,
			generate_trajectories: bool = True,
			force_generate_trajectories: bool = False,
			verbose: bool = True,
	):
		self.name = name
		self.mass = mass
		self.noise = noise
		self.engine_force = engine_force
		super().__init__(
			env=env,
			name=name,
			n_trajectories=n_trajectories,
			trajectories_folder=trajectories_folder,
			generate_trajectories=generate_trajectories,
			force_generate_trajectories=force_generate_trajectories,
			verbose=verbose,
		)

	def forward(self, inputs, **kwargs):
		shape = (inputs[0].shape[0], self.spec.action_spec.continuous_size)
		return np.random.normal(0.99 * self.mass * 9.81, self.noise, size=shape) / self.engine_force


class RandomHeuristic(FgHeuristic):
	def __init__(self, env: BaseEnv, name: str = "random_heuristic", **kwargs):
		super().__init__(env, name, **kwargs)

	def get_actions(self, inputs, **kwargs) -> ActionTuple:
		return self.spec.action_spec.random_action(inputs[0].shape[0])







