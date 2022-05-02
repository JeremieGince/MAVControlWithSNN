import collections.abc
import os
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple
# from PythonAcademy.models.dqn import DQN, dqn_loss
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from PythonAcademy.src.curriculum import Curriculum
from PythonAcademy.src.wrappers import TensorActionTuple


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


class TrainingHistory:
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

	def min(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			return min(self[key])
		return np.inf

	def min_item(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			argmin = np.argmin(self[key])
			return {k: v[argmin] for k, v in self.items()}
		raise ValueError("key not in container")

	def max(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			return max(self[key])
		return -np.inf

	def max_item(self, key=None):
		if key is None:
			key = list(self.container.keys())[0]
		if key in self:
			argmax = np.argmax(self[key])
			return {k: v[argmax] for k, v in self.items()}
		raise ValueError("key not in container")

	@staticmethod
	def _set_default_plot_kwargs(kwargs: dict):
		kwargs.setdefault('fontsize', 16)
		kwargs.setdefault('linewidth', 3)
		kwargs.setdefault('figsize', (16, 12))
		kwargs.setdefault('dpi', 300)
		return kwargs

	def plot(
			self,
			save_path=None,
			show=False,
			**kwargs
	):
		kwargs = self._set_default_plot_kwargs(kwargs)
		loss_metrics = [k for k in self.container if 'loss' in k.lower()]
		other_metrics = [k for k in self.container if k not in loss_metrics]
		n_cols = int(np.sqrt(1 + len(other_metrics)))
		n_rows = int((1 + len(other_metrics)) / n_cols)
		fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=kwargs["figsize"], sharex='all')
		axes = np.ravel(axes)
		for i, ax in enumerate(axes):
			if i == 0:
				for k in loss_metrics:
					ax.plot(self[k], label=k, linewidth=kwargs['linewidth'])
				ax.set_ylabel("Loss [-]", fontsize=kwargs["fontsize"])
				ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				ax.legend(fontsize=kwargs["fontsize"])
			else:
				k = other_metrics[i - 1]
				ax.plot(self[k], label=k, linewidth=kwargs['linewidth'])
				ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				ax.legend(fontsize=kwargs["fontsize"])
		if save_path is not None:
			plt.savefig(save_path, dpi=kwargs["dpi"])
		if show:
			plt.show()
		plt.close(fig)


class TrainingHistoriesMap:
	REPORT_KEY = "report"

	def __init__(self, curriculum: Optional[Curriculum] = None):
		self.curriculum = curriculum
		self.histories = defaultdict(TrainingHistory, **{TrainingHistoriesMap.REPORT_KEY: TrainingHistory()})

	@property
	def report_history(self) -> TrainingHistory:
		return self.histories[TrainingHistoriesMap.REPORT_KEY]

	def max(self, key=None):
		if self.curriculum is None:
			return self.histories[TrainingHistoriesMap.REPORT_KEY].max(key)
		else:
			return self.histories[self.curriculum.current_lesson.name].max(key)

	def concat(self, other):
		self.histories[TrainingHistoriesMap.REPORT_KEY].concat(other)
		if self.curriculum is not None:
			return self.histories[self.curriculum.current_lesson.name].concat(other)

	def append(self, key, value):
		self.histories[TrainingHistoriesMap.REPORT_KEY].append(key, value)
		if self.curriculum is not None:
			return self.histories[self.curriculum.current_lesson.name].append(key, value)

	@staticmethod
	def _set_default_plot_kwargs(kwargs: dict):
		kwargs.setdefault('fontsize', 16)
		kwargs.setdefault('linewidth', 3)
		kwargs.setdefault('figsize', (16, 12))
		kwargs.setdefault('dpi', 300)
		return kwargs

	def plot(self, save_path=None, show=False, lesson_idx: Optional[Union[int, str]] = None, **kwargs):
		kwargs = self._set_default_plot_kwargs(kwargs)
		if self.curriculum is None:
			assert lesson_idx is None, "lesson_idx must be None if curriculum is None"
			return self.plot_history(TrainingHistoriesMap.REPORT_KEY, save_path, show, **kwargs)
		if lesson_idx is None:
			self.plot_history(TrainingHistoriesMap.REPORT_KEY, save_path, show, **kwargs)
		else:
			self.plot_history(self.curriculum[lesson_idx].name, save_path, show, **kwargs)

	def plot_history(
			self,
			history_name: str,
			save_path=None,
			show=False,
			**kwargs
	):
		os.makedirs(os.path.dirname(save_path), exist_ok=True)
		history = self.histories[history_name]
		if self.curriculum is not None and history_name != TrainingHistoriesMap.REPORT_KEY:
			lessons = [self.curriculum[history_name]]
			lessons_start_itr = [0]
		elif self.curriculum is not None and history_name == TrainingHistoriesMap.REPORT_KEY:
			lessons = self.curriculum.lessons
			lessons_lengths = {k: [len(self.histories[lesson.name][k]) for lesson in lessons] for k in history.container}
			lessons_start_itr = {k: np.cumsum(lessons_lengths[k]) for k in history.container}
		else:
			lessons = []
			lessons_start_itr = []

		kwargs = self._set_default_plot_kwargs(kwargs)
		loss_metrics = [k for k in history.container if 'loss' in k.lower()]
		rewards_metrics = [k for k in history.container if 'reward' in k.lower()]
		other_metrics = [k for k in history.container if k not in loss_metrics and k not in rewards_metrics]
		n_metrics = 2 + len(other_metrics)
		n_cols = int(np.sqrt(n_metrics))
		n_rows = int(n_metrics / n_cols)
		fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=kwargs["figsize"], sharex='all')
		if axes.ndim == 1:
			axes = np.expand_dims(axes, axis=-1)
		for row_i in range(n_rows):
			for col_i in range(n_cols):
				ax = axes[row_i, col_i]
				ravel_index = row_i * n_cols + col_i
				if ravel_index == 0:
					for k in loss_metrics:
						ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
					ax.set_ylabel("Loss [-]", fontsize=kwargs["fontsize"])
					ax.legend(fontsize=kwargs["fontsize"])
				elif ravel_index == 1:
					for k in rewards_metrics:
						ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
						for lesson_idx, lesson in enumerate(lessons):
							if lesson.completion_criteria.measure == k:
								ax.plot(
									lesson.completion_criteria.threshold*np.ones(len(history[k])), 'k--',
									label=f"{k} threshold", linewidth=kwargs['linewidth']
								)
							if history_name == TrainingHistoriesMap.REPORT_KEY and lesson.is_completed:
								ax.axvline(
									lessons_start_itr[k][lesson_idx], ymin=np.min(history[k]), ymax=np.max(history[k]),
									color='r', linestyle='--', linewidth=kwargs['linewidth'], label=f"lesson start"
								)
					ax.set_ylabel("Rewards [-]", fontsize=kwargs["fontsize"])
					ax.legend(fontsize=kwargs["fontsize"])
				else:
					k = other_metrics[ravel_index - 1]
					ax.plot(history[k], label=k, linewidth=kwargs['linewidth'])
					ax.legend(fontsize=kwargs["fontsize"])
				if row_i == n_rows - 1:
					ax.set_xlabel("Iterations [-]", fontsize=kwargs["fontsize"])
				legend_without_duplicate_labels_(ax)
		if save_path is not None:
			plt.savefig(save_path, dpi=kwargs["dpi"])
		if show:
			plt.show()
		plt.close(fig)


def to_tensor(x, dtype=torch.float32):
	if isinstance(x, np.ndarray):
		return torch.from_numpy(x).type(dtype)
	elif not isinstance(x, torch.Tensor):
		return torch.tensor(x, dtype=dtype)
	return x.type(dtype)


def linear_decay(init_value, min_value, decay_value, current_itr):
	return max(init_value * decay_value ** current_itr, min_value)


def send_parameter_to_channel(
		channel: EnvironmentParametersChannel,
		parameters: Dict[str, Any]
) -> Dict[str, float]:
	"""
	Convert a dictionary of parameters to a dictionary of floats and send it to the channel.
	:param channel: The channel to send the parameters to.
	:param parameters: The parameters to send. Each value should be able to be converted to a float.
	:return: The parameters as floats.
	"""
	float_params = {k: float(v) for k, v in parameters.items()}
	for key, value in float_params.items():
		channel.set_float_parameter(key, value)
	return float_params


def threshold_image(image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
	if isinstance(image, np.ndarray):
		return np.where(image > 0.5, 1.0, 0.0).astype(image.dtype)
	elif isinstance(image, torch.Tensor):
		return torch.where(image > 0.5, torch.ones_like(image), torch.zeros_like(image)).type(image.dtype)
	else:
		raise ValueError("image must be a numpy array or a torch tensor")


def unbatch_actions(
		actions: Union[ActionTuple, TensorActionTuple]
) -> List[Union[ActionTuple, TensorActionTuple]]:
	"""
	:param actions: shape: (batch_size, ...)
	:return:
	"""
	dtype = type(actions)
	actions_list = []
	continuous, discrete = actions.continuous, actions.discrete
	batch_size = actions.continuous.shape[0] if continuous is not None else discrete.shape[0]
	assert batch_size is not None
	for i in range(batch_size):
		actions_list.append(
			dtype(
				continuous[i] if continuous is not None else None,
				discrete[i] if discrete is not None else None)
		)
	return actions_list


def discount_rewards(r, gamma=0.99, value_next=0.0):
	"""
	Computes discounted sum of future rewards for use in updating value estimate.
	:param r: List of rewards.
	:param gamma: Discount factor.
	:param value_next: T+1 value estimate for returns calculation.
	:return: discounted sum of future rewards as list.
	"""
	discounted_r = np.zeros_like(r)
	running_add = value_next
	for t in reversed(range(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r


def compute_advantage(rewards, value_estimates, value_next=0.0, gamma=0.99, lambd=0.95):
	"""
	Computes generalized advantage estimate for use in updating policy.
	:param rewards: list of rewards for time-steps t to T.
	:param value_next: Value estimate for time-step T+1.
	:param value_estimates: list of value estimates for time-steps t to T.
	:param gamma: Discount factor.
	:param lambd: GAE weighing factor.
	:return: list of advantage estimates for time-steps t to T.
	"""
	value_estimates = np.append(value_estimates, value_next)
	delta_t = rewards + gamma * value_estimates[1:] - value_estimates[:-1]
	advantage = discount_rewards(r=delta_t, gamma=gamma * lambd)
	return advantage


def legend_without_duplicate_labels_(ax: plt.Axes):
	handles, labels = ax.get_legend_handles_labels()
	unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
	ax.legend(*zip(*unique))

