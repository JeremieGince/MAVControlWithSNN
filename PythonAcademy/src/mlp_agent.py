from typing import Callable, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple, BehaviorSpec, DimensionProperty
from torch import nn

from PythonAcademy.src.base_agent import BaseAgent
from PythonAcademy.src.utils import TensorActionTuple


class MLPAgent(BaseAgent):
	def __init__(
			self,
			spec: BehaviorSpec,
			behavior_name: str,
			n_hidden_neurons: Union[Iterable[int], int] = None,
			checkpoint_folder: str = "checkpoints",
			name: str = "mlp",
			device: torch.device = None,
			input_transform: Union[Dict[str, Callable], List[Callable]] = None,
			**kwargs
	):
		super().__init__(
			spec=spec,
			behavior_name=behavior_name,
			name=name,
			checkpoint_folder=checkpoint_folder,
			device=device,
			input_transform=input_transform,
			**kwargs
		)
		if isinstance(n_hidden_neurons, int):
			n_hidden_neurons = [n_hidden_neurons]
		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else [128, 128]
		if self.n_hidden_neurons:
			self.first_hidden_sizes = self._dispatch_first_hidden_size()

		self.input_layers = nn.ModuleDict()
		self.layers = nn.ModuleDict()
		self._add_layers_()
		self.all_layer_names = list(self.input_layers.keys()) + list(self.layers.keys())

	@property
	def input_sizes(self) -> Dict[str, int]:
		return {
			obs.name: np.prod([
				d
				for d, d_type in zip(obs.shape, obs.dimension_property)
			], dtype=int)
			for obs in self.spec.observation_specs
		}

	@property
	def output_size(self) -> int:
		return int(self.spec.action_spec.continuous_size + self.spec.action_spec.discrete_size)

	def _dispatch_first_hidden_size(self) -> Dict[str, int]:
		norm_factor = np.sum([v for _, v in self.input_sizes.items()])
		ratios = {k: v / norm_factor for k, v in self.input_sizes.items()}
		first_hidden_sizes = {k: int(v * self.n_hidden_neurons[0]) for k, v in ratios.items()}
		if self.n_hidden_neurons[0] != sum(first_hidden_sizes.values()):
			key_max_size = max(self.input_sizes, key=lambda k: self.input_sizes[k])
			first_hidden_sizes[key_max_size] += self.n_hidden_neurons[0] - sum(first_hidden_sizes.values())
		return first_hidden_sizes

	def _add_input_layers_(self):
		if not self.n_hidden_neurons:
			return
		for l_name, in_size in self.input_sizes.items():
			self.input_layers[l_name] = nn.Sequential(
				nn.Flatten(),
				nn.Linear(
					in_features=in_size,
					out_features=self.first_hidden_sizes[l_name],
					device=self.device,
				),
				nn.ReLU(),
			)

	def _add_hidden_layers_(self):
		if not self.n_hidden_neurons:
			return
		for i, hn in enumerate(self.n_hidden_neurons[:-1]):
			self.layers[f"hidden_{i}"] = nn.Sequential(
				nn.Flatten(),
				nn.Linear(
					in_features=hn,
					out_features=self.n_hidden_neurons[i + 1],
					device=self.device,
				),
				nn.ReLU(),
			)

	def _add_readout_layer(self):
		if self.n_hidden_neurons:
			in_size = self.n_hidden_neurons[-1]
		else:
			in_size = int(np.prod(self.input_sizes.values()))
		self.layers["readout"] = nn.Sequential(
			nn.Flatten(),
			nn.Linear(
				in_features=in_size,
				out_features=self.output_size,
				device=self.device,
			),
			# nn.BatchNorm1d(self.output_size, device=self.device),
		)

	def _add_layers_(self):
		self._add_input_layers_()
		self._add_hidden_layers_()
		self._add_readout_layer()

	def forward(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]],
			**kwargs
	) -> torch.Tensor:
		inputs = self.apply_transform(obs)

		features_list = [
			layer(inputs[layer_name])
			for layer_name, layer in self.input_layers.items()
		]
		if features_list:
			forward_tensor = torch.concat(features_list, dim=1)
		else:
			forward_tensor = torch.concat([inputs[in_name] for in_name in inputs], dim=1)

		for layer_idx, (layer_name, layer) in enumerate(self.layers.items()):
			forward_tensor = layer(forward_tensor)
		return forward_tensor

	def get_logits(self, obs: Sequence[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
		return self(obs)

	def _obs_forward_to_logits(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]]
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:param obs: shape: (observation_size, nb_agents, )
		:return:
		"""
		outputs = self(obs)
		continuous_outputs = outputs[:, :self.spec.action_spec.continuous_size]
		discrete_outputs = []
		if self.spec.action_spec.discrete_size > 0:
			discrete_values = outputs[:, self.spec.action_spec.continuous_size:]
			for branch_idx, branch in enumerate(self.spec.action_spec.discrete_branches):
				branch_cum_idx = sum(self.spec.action_spec.discrete_branches[:branch_idx])
				branch_values = discrete_values[:, branch_cum_idx:branch_cum_idx + branch]
				discrete_outputs.append(torch.max(branch_values, dim=-1).values)
		if discrete_outputs:
			discrete_outputs = torch.cat(discrete_outputs, dim=-1)
		return continuous_outputs, discrete_outputs

	def get_actions(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]],
			**kwargs
	) -> TensorActionTuple:
		"""
		Get the actions for the given observations.
		:param obs: The observations.
		False if the actions should be returned as torch tensors, used for training.
		:param kwargs: Other arguments.
		:return: The actions.
		"""
		continuous_outputs, discrete_outputs = self._obs_forward_to_logits(obs)
		return TensorActionTuple(continuous_outputs, discrete_outputs)
