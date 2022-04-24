import collections.abc
import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import torch

# from PythonAcademy.models.dqn import DQN, dqn_loss


def set_random_seed(environment, seed):
	environment.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)


# def load_model(model_type, path_weights, environment, memory_size=20, model_kwargs={}):
# 	model = model_type(environment.observation_space.shape,
# 	                   environment.action_space.n,
# 	                   memory_size=memory_size,
# 	                   **model_kwargs)
# 	m = DQN(
# 		list(range(environment.action_space.n)),
# 		model,
# 		optimizer="sgd",
# 		loss_function=dqn_loss,
# 	)
# 	m.load_weights(path_weights)
# 	return m


def show_rewards(R: Iterable, **kwargs):
	plt.plot(R)
	plt.yticks(np.arange(max(np.min(R), -200), np.max(R) + 1, 50))
	plt.grid()
	title = kwargs.get("title", "Reward per episodes")
	plt.title(title)
	plt.ylabel("Reward [-]")
	plt.xlabel("Episodes [-]")

	subfolder = kwargs.get("subfolder", False)
	if subfolder:
		os.makedirs(f"figures/{subfolder}/", exist_ok=True)
		plt.savefig(f"figures/{subfolder}/Projet_{title.replace(' ', '_').replace(':', '_')}.png", dpi=300)
	else:
		os.makedirs("RNN/figures/", exist_ok=True)
		plt.savefig(f"figures/Projet_{title.replace(' ', '_').replace(':', '_')}.png", dpi=300)
	plt.show(block=kwargs.get("block", True))


def batchwise_temporal_filter(x: torch.Tensor, decay: float = 0.9):
	"""
	:param x: (batch_size, time_steps, ...)
	:param decay:
	:return:
	"""
	batch_size, time_steps, *_ = x.shape
	assert time_steps >= 1

	powers = torch.arange(time_steps, dtype=torch.float32, device=x.device).flip(0)
	weighs = torch.pow(decay, powers)

	x = torch.mul(x, weighs.unsqueeze(0).unsqueeze(-1))
	x = torch.sum(x, dim=1)
	return x


def mapping_update_recursively(d, u):
	"""
	from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
	:param d: mapping item that wil be updated
	:param u: mapping item updater
	:return: updated mapping recursively
	"""
	for k, v in u.items():
		if isinstance(v, collections.abc.Mapping):
			d[k] = mapping_update_recursively(d.get(k, {}), v)
		else:
			d[k] = v
	return d


class LossHistory:
	def __init__(self, container: Dict[str, List[float]] = None):
		self.container = defaultdict(list)
		if container is not None:
			self.container.update(container)

	def __getitem__(self, item):
		return self.container[item]

	def __setitem__(self, key, value):
		self.container[key] = value

	def __contains__(self, item):
		return item in self.container

	def __iter__(self):
		return iter(self.container)

	def __len__(self):
		return len(self.container)

	def items(self):
		return self.container.items()

	def concat(self, other):
		for key, values in other.items():
			if isinstance(values, list):
				self.container[key].extend(values)
			else:
				self.container[key].append(values)

	def append(self, key, value):
		self.container[key].append(value)

	def min(self, key='val'):
		if key in self:
			return min(self[key])
		return np.inf

	def min_item(self, key='val'):
		if key in self:
			argmin = np.argmin(self[key])
			return {k: v[argmin] for k, v in self.items()}

	def plot(self, save_path=None, show=False):
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots(figsize=(12, 10))
		for name, values in self.items():
			ax.plot(values, label=name, linewidth=3)
		ax.set_xlabel("Epoch [-]", fontsize=16)
		ax.set_ylabel("Loss [-]", fontsize=16)
		ax.legend(fontsize=16)
		if save_path is not None:
			plt.savefig(save_path, dpi=300)
		if show:
			plt.show()
		plt.close(fig)


def to_tensor(x, dtype=torch.float32):
	if isinstance(x, np.ndarray):
		return torch.from_numpy(x).type(dtype)
	elif not isinstance(x, torch.Tensor):
		return torch.tensor(x, dtype=dtype)
	return x.type(dtype)

