import pickle
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
	def __init__(self, capacity, seed=None):
		self.__capacity = capacity
		self.random_generator = np.random.RandomState(seed)
		self.data: List[Experience] = []
		self._counter = 0
		self._counter_is_started = False

	@property
	def counter(self):
		return self._counter

	@property
	def capacity(self):
		return self.__capacity
	
	def set_seed(self, seed: int):
		self.random_generator.seed(seed)

	def start_counter(self):
		self._counter_is_started = True
		self._counter = 0

	def stop_counter(self):
		self._counter_is_started = False
		self._counter = 0

	def reset_counter(self):
		self.stop_counter()

	def increment_counter(self, increment: int = 1):
		self._counter += increment

	def increase_capacity(self, increment: int):
		self.__capacity += increment
	
	def extend(self, iterable: Iterable[Experience]):
		_ = [self.store(e) for e in iterable]
	
	def __len__(self):
		return len(self.data)

	def __iter__(self):
		return iter(self.data)

	def __getitem__(self, idx: int) -> Experience:
		return self.data[idx]
	
	def store(self, element: Experience):
		"""
		Stores an element. If the replay buffer is already full, deletes the oldest
		element to make space.
		"""
		if len(self.data) >= self.__capacity:
			self.data.pop(0)
		self.data.append(element)
		if self._counter_is_started:
			self._counter += 1
	
	def get_random_batch(self, batch_size: int) -> List[Experience]:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		return self.random_generator.choice(self.data, size=batch_size)
	
	def get_batch_tensor(self, batch_size: int, device='cpu') -> BatchExperience:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		batch = self.get_random_batch(batch_size)
		return self._make_batch(batch, device=device)
	
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
			n_batches: int = None,
			randomize: bool = True,
			device='cpu',
	) -> Iterable[BatchExperience]:
		"""
		Returns a generator of batch_size elements from the buffer.
		"""
		max_idx = int(batch_size * int(len(self) / batch_size))
		indexes = np.arange(max_idx).reshape(-1, batch_size)
		if n_batches is None:
			n_batches = indexes.shape[0]
		else:
			n_batches = min(n_batches, indexes.shape[0])
		if randomize:
			self.random_generator.shuffle(indexes)
		for i in range(n_batches):
			batch = [self.data[j] for j in indexes[i]]
			yield self._make_batch(batch, device=device)

	def save(self, filename: str):
		with open(filename, 'wb') as file:
			pickle.dump(self, file)

	@staticmethod
	def load(filename: str) -> 'ReplayBuffer':
		with open(filename, 'rb') as file:
			buffer = pickle.load(file)
		return buffer
