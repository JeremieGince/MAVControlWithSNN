from typing import Any, Iterable, List, NamedTuple

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple


class Experience(NamedTuple):
	"""
	An experience contains the data of one Agent transition.
	- Observation
	- Action
	- Reward
	- Done flag
	- Next Observation
	"""
	obs: Any
	action: ActionTuple
	reward: float
	terminal: bool
	next_obs: Any


class BatchExperience(NamedTuple):
	"""
	A batch of experiences.
	- Observation
	- Action
	- Reward
	- Done flag
	- Next Observation
	"""
	obs: List[torch.Tensor]
	continuous_actions: torch.Tensor
	discrete_actions: torch.Tensor
	rewards: torch.Tensor
	terminals: torch.Tensor
	next_obs: List[torch.Tensor]


Trajectory = List[Experience]


class ReplayBuffer:
	def __init__(self, buffer_size, seed=None):
		self.__buffer_size = buffer_size
		self.random_generator = np.random.RandomState(seed)
		self.data: List[Experience] = []
	
	def set_seed(self, seed: int):
		self.random_generator.seed(seed)
	
	def extend(self, iterable: Iterable):
		_ = [self.store(e) for e in iterable]
	
	def __len__(self):
		return len(self.data)
	
	def store(self, element: Experience):
		"""
		Stores an element. If the replay buffer is already full, deletes the oldest
		element to make space.
		"""
		if len(self.data) >= self.__buffer_size:
			self.data.pop(0)
		
		self.data.append(element)
	
	def get_random_batch(self, batch_size: int) -> List[Experience]:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		return self.random_generator.choice(self.data, size=batch_size)
	
	def get_batch_tensor(self, batch_size: int) -> BatchExperience:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		batch = self.get_random_batch(batch_size)
		return self._make_batch(batch)
	
	@staticmethod
	def _make_batch(batch: List[Experience], device='cpu') -> BatchExperience:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		nb_obs = len(batch[0].obs)
		obs = [torch.from_numpy(np.stack([ex.obs[i] for ex in batch])).to(device) for i in range(nb_obs)]
		rewards = torch.from_numpy(
			np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
		).to(device)
		terminals = torch.from_numpy(
			np.array([ex.terminal for ex in batch], dtype=np.float32).reshape(-1, 1)
		).to(device)
		continuous_actions = torch.from_numpy(np.stack([ex.action.continuous for ex in batch])).to(device)
		discrete_actions = torch.from_numpy(np.stack([ex.action.discrete for ex in batch])).to(device)
		next_obs = [torch.from_numpy(np.stack([ex.next_obs[i] for ex in batch])).to(device) for i in range(nb_obs)]
		return BatchExperience(
			obs=obs,
			continuous_actions=continuous_actions,
			discrete_actions=discrete_actions,
			rewards=rewards,
			terminals=terminals,
			next_obs=next_obs,
		)
	
	def get_batch_generator(
			self,
			batch_size: int,
			randomize: bool = True,
			device='cpu',
	) -> Iterable[BatchExperience]:
		"""
		Returns a generator of batch_size elements from the buffer.
		"""
		indexes = np.arange(len(self)).reshape(-1, batch_size)
		if randomize:
			self.random_generator.shuffle(indexes)
		for batch_indexes in indexes:
			batch = [self.data[i] for i in batch_indexes]
			yield self._make_batch(batch, device=device)
