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
			name: str = "heuristic1DF",
			mass: float = 1.0,
			noise: float = 0.1,
			engine_force: float = 20.0,
			n_trajectories: int = 1024,
			trajectories_folder: str = None,
			generate_trajectories: bool = True,
			verbose: bool = True,
	):
		self.name = name
		self.mass = mass
		self.noise = noise
		self.engine_force = engine_force
		self.n_trajectories = n_trajectories
		self.verbose = verbose
		if trajectories_folder is None:
			trajectories_folder = self.name
		self.trajectories_folder = trajectories_folder
		assert '.' not in self.trajectories_folder, "Folder name should not contain '.'"
		self._trajectories = self.load_trajectories()
		if generate_trajectories and len(self._trajectories) < self.n_trajectories:
			if self._trajectories.capacity < self.n_trajectories:
				self._trajectories.increase_capacity(self.n_trajectories - self._trajectories.capacity)
			self.generate_trajectories(env, verbose)
			self.save_trajectories()

	@property
	def trajectories(self) -> ReplayBuffer:
		return self._trajectories

	@property
	def buffer(self) -> ReplayBuffer:
		return self._trajectories

	def forward(self, inputs, **kwargs):
		return np.random.normal(0.99 * self.mass * 9.81, self.noise, size=inputs[0].shape[0]) / self.engine_force

	def get_actions(self, inputs, **kwargs) -> ActionTuple:
		return ActionTuple(continuous=self.forward(inputs), discrete=None)

	def __call__(self, *args, **kwargs) -> ActionTuple:
		return self.get_actions(*args, **kwargs)

	def save_trajectories(self):
		self._trajectories.save(f"{self.trajectories_folder}/buffer.pkl")

	def load_trajectories(self):
		if not os.path.exists(f"{self.trajectories_folder}/buffer.pkl"):
			return ReplayBuffer(self.n_trajectories)
		buffer = ReplayBuffer.load(f"{self.trajectories_folder}/buffer.pkl")
		self._trajectories = buffer
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
		p_bar = tqdm(total=self.n_trajectories, disable=not verbose, desc=f"Generating Trajectories - {self.name}")
		last_buffer_counter = 0
		while len(self._trajectories) < self.n_trajectories:  # While not enough data in the buffer
			decision_steps, terminal_steps = env.get_steps(behavior_name)
			cumulative_rewards.extend(SNNAgent.exec_terminal_steps_(agent_maps, self._trajectories, terminal_steps))
			if len(decision_steps) > 0:
				SNNAgent.exec_decisions_steps_(agent_maps, decision_steps)
				actions = self(decision_steps.obs)
				actions_list = SNNAgent.unbatch_actions(actions)
				for agent_index, agent_id in enumerate(decision_steps.agent_id):
					agent_maps.last_action[agent_id] = actions_list[agent_index]
				env.set_actions(behavior_name, actions)
			p_bar.update(len(self._trajectories) - last_buffer_counter)
			last_buffer_counter = len(self._trajectories)
			env.step()
		p_bar.close()
		return self._trajectories, cumulative_rewards








