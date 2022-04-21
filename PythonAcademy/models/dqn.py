import random
from typing import Any, Iterable, List, NamedTuple, Tuple, Union

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple, BehaviorSpec
from poutyne import Model


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
	done: bool
	next_obs: Any


Trajectory = List[Experience]


class ReplayBuffer:
	def __init__(self, buffer_size):
		self.__buffer_size = buffer_size
		self.data: List[Experience] = []

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

	def get_batch(self, batch_size: int) -> List[Experience]:
		"""
		Returns a list of batch_size elements from the buffer.
		"""
		return random.choices(self.data, k=batch_size)


class DQNAgent(torch.nn.Module):
	def __init__(self, spec: BehaviorSpec):
		super(DQNAgent, self).__init__()
		self.spec = spec

	def forward(self, obs):
		raise NotImplementedError()

	def soft_update(self, other, tau):
		"""
		Code for the soft update between a target network (self) and
		a source network (other).

		The weights are updated according to the rule in the assignment.
		"""
		new_weights = {}

		own_weights = self.get_weight_copies()
		other_weights = other.get_weight_copies()

		for k in own_weights:
			new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

		self.set_weights(new_weights)


class DQN(Model):
	def __init__(self, actions, *args, **kwargs):
		self.actions = actions
		super().__init__(*args, **kwargs)

	def __call__(self, *args, **kwargs):
		return self.get_action(*args, **kwargs)

	def get_action(self, state: Union[torch.Tensor, np.ndarray], epsilon: float) -> int:
		"""
		Returns the selected action according to an epsilon-greedy policy.
		"""
		if np.random.random() < epsilon:
			return np.random.choice(self.actions)
		else:
			return np.argmax(self.predict([
				state[np.newaxis, ],
			], batch_size=1).squeeze()).item()

	def soft_update(self, other, tau):
		"""
		Code for the soft update between a target network (self) and
		a source network (other).

		The weights are updated according to the rule in the assignment.
		"""
		new_weights = {}

		own_weights = self.get_weight_copies()
		other_weights = other.get_weight_copies()

		for k in own_weights:
			new_weights[k] = (1 - tau) * own_weights[k] + tau * other_weights[k]

		self.set_weights(new_weights)


def format_batch(
		batch: List[Tuple[np.ndarray, int, float, np.ndarray, bool]],
		target_network: DQN,
		gamma: float
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray, np.ndarray]]:
	"""
	Input :
		- batch, a list of n=batch_size elements from the replay buffer
		- target_network, the target network to compute the one-step lookahead target
		- gamma, the discount factor

	Returns :
		- states, a numpy array of size (batch_size, state_dim) containing the states in the batch
		- (actions, targets) : where actions and targets both
					  have the shape (batch_size, ). Actions are the
					  selected actions according to the target network
					  and targets are the one-step lookahead targets.
	"""
	states = np.array([e[0] for e in batch])
	actions = np.array([e[1] for e in batch])

	next_states = np.array([e[3] for e in batch])
	target_predictions = target_network.predict([next_states, ], batch_size=len(next_states))
	targets = np.array([e[2] + gamma * np.max(q) * (1 - e[4]) for e, q in zip(batch, target_predictions)])
	return (states,), (actions, targets)


def dqn_loss(y_pred: torch.Tensor, y_target: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
	"""
	Input :
		- y_pred, (batch_size, n_actions) Tensor outputted by the network
		- y_target = (actions, targets), where actions and targets both
					  have the shape (batch_size, ). Actions are the
					  selected actions according to the target network
					  and targets are the one-step lookahead targets.

	Returns :
		- The DQN loss
	"""
	actions, targets = y_target
	return torch.mean(torch.pow(targets.detach() - y_pred[np.arange(y_pred.shape[0]), actions.long()], 2))
