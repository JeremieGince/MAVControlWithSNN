import json
import os
import shutil
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from mlagents_envs.base_env import ActionTuple, BaseEnv, BehaviorSpec, DecisionSteps, DimensionProperty, TerminalSteps
from mlagents_envs.environment import UnityEnvironment
from torch import Tensor, nn
from torchvision.transforms import Compose, Lambda
from tqdm.auto import tqdm

from PythonAcademy.src.base_agent import BaseAgent
from PythonAcademy.src.buffers import BatchExperience, Experience, ReplayBuffer
from PythonAcademy.src.curriculum import Curriculum
from PythonAcademy.src.rl_academy import AgentsHistoryMaps, LoadCheckpointMode
from PythonAcademy.src.spike_funcs import HeavisideSigmoidApprox, SpikeFuncType, SpikeFuncType2Func, SpikeFunction
from PythonAcademy.src.spiking_layers import ALIFLayer, LIFLayer, LayerType, LayerType2Layer, ReadoutLayer
from PythonAcademy.src.utils import TrainingHistory, linear_decay, mapping_update_recursively, to_tensor
from PythonAcademy.src.wrappers import TensorActionTuple


class SNNAgent(BaseAgent):
	def __init__(
			self,
			spec: BehaviorSpec,
			behavior_name: str,
			n_hidden_neurons: Union[Iterable[int], int] = None,
			use_recurrent_connection: Union[bool, Iterable[bool]] = True,
			int_time_steps: int = None,
			spike_func: Union[Type[SpikeFunction], SpikeFuncType] = HeavisideSigmoidApprox,
			hidden_layer_type: Union[Type[LIFLayer], LayerType] = ALIFLayer,
			checkpoint_folder: str = "checkpoints",
			name: Optional[str] = None,
			device: torch.device = None,
			input_transform: Union[Dict[str, Callable], List[Callable]] = None,
			**kwargs
	):
		"""
		Initialize the SNN agent.
		:param spec: The behavior spec of the agent.
		The spec is used to determine the number of actions and observations.
		:param behavior_name: The name of the behavior.
		:param n_hidden_neurons: The number of hidden neurons in the network.
		:param use_recurrent_connection: True if recurrent connections should be used.
		:param int_time_steps: The number of time steps to use for the internal representation.
		:param spike_func: The spike function to use.
		:param hidden_layer_type: The type of hidden layer to use.
		:param device: The device to use.
		:param checkpoint_folder: The folder to save the checkpoints in.
		:param name: The name of the model.
		:param input_transform: The transform to apply to the observations.
		:param kwargs: The other arguments.
		"""
		if name is None:
			name = f"{hidden_layer_type.__name__}_snn"
		super(SNNAgent, self).__init__(
			spec=spec,
			behavior_name=behavior_name,
			name=name,
			checkpoint_folder=checkpoint_folder,
			device=device,
			input_transform=input_transform,
			**kwargs
		)
		self.dt = self.kwargs.get("dt", 1e-3)
		self.int_time_steps = self._get_and_check_int_time_steps() if int_time_steps is None else int_time_steps
		if isinstance(spike_func, SpikeFuncType):
			spike_func = SpikeFuncType2Func[spike_func]
		self.spike_func = spike_func
		if isinstance(hidden_layer_type, LayerType):
			hidden_layer_type = LayerType2Layer[hidden_layer_type]
		self.hidden_layer_type = hidden_layer_type

		if isinstance(n_hidden_neurons, int):
			n_hidden_neurons = [n_hidden_neurons]
		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else []
		if self.n_hidden_neurons:
			self.first_hidden_sizes = self._dispatch_first_hidden_size()
		self.use_recurrent_connection = use_recurrent_connection
		self.input_layers = nn.ModuleDict()
		self.layers = nn.ModuleDict()
		self._add_layers_()
		self.all_layer_names = list(self.input_layers.keys()) + list(self.layers.keys())
		self.initialize_weights_()

	@property
	def input_sizes(self) -> Dict[str, int]:
		return {
			obs.name: np.prod([
				d
				for d, d_type in zip(obs.shape, obs.dimension_property)
				if d_type == DimensionProperty.TRANSLATIONAL_EQUIVARIANCE
			], dtype=int)
			for obs in self.spec.observation_specs
		}

	@property
	def output_size(self) -> int:
		return int(self.spec.action_spec.continuous_size + self.spec.action_spec.discrete_size)

	def _get_and_check_int_time_steps(self) -> int:
		int_times = [
			d
			for obs in self.spec.observation_specs
			for d, d_type in zip(obs.shape, obs.dimension_property)
			if d_type == DimensionProperty.NONE
		]
		assert len(set(int_times)) == 1, "All int_times must be the same"
		return int_times[0]

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
			self.input_layers[l_name] = self.hidden_layer_type(
				input_size=in_size,
				output_size=self.first_hidden_sizes[l_name],
				use_recurrent_connection=self.use_recurrent_connection,
				dt=self.dt,
				spike_func=self.spike_func,
				device=self.device,
				**self.kwargs
			)

	def _add_hidden_layers_(self):
		if not self.n_hidden_neurons:
			return
		for i, hn in enumerate(self.n_hidden_neurons[:-1]):
			self.layers[f"hidden_{i}"] = self.hidden_layer_type(
				input_size=hn,
				output_size=self.n_hidden_neurons[i + 1],
				use_recurrent_connection=self.use_recurrent_connection,
				dt=self.dt,
				spike_func=self.spike_func,
				device=self.device,
				**self.kwargs
			)

	def _add_readout_layer(self):
		if self.n_hidden_neurons:
			in_size = self.n_hidden_neurons[-1]
		else:
			in_size = int(np.prod(self.input_sizes.values()))
		self.layers["readout"] = ReadoutLayer(
			input_size=in_size,
			output_size=self.output_size,
			dt=self.dt,
			spike_func=self.spike_func,
			device=self.device,
			**self.kwargs
		)

	def _add_layers_(self):
		self._add_input_layers_()
		self._add_hidden_layers_()
		self._add_readout_layer()

	def initialize_weights_(self):
		for param in self.parameters():
			if param.ndim > 2:
				torch.nn.init.xavier_normal_(param)
			else:
				torch.nn.init.normal_(param)
		for layer_name, layer in self.layers.items():
			if getattr(layer, "initialize_weights_") and callable(layer.initialize_weights_):
				layer.initialize_weights_()

	def _format_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
		"""
		Check the shape of the inputs. If the shape of the inputs is (batch_size, features),
		the inputs is considered constant over time and the inputs will be repeat over self.int_time_steps time steps.
		If the shape of the inputs is (batch_size, time_steps, features), time_steps must be less are equal to
		self.int_time_steps and the inputs will be padded by zeros for time steps greater than time_steps.
		:param inputs: Inputs tensor
		:return: Formatted Input tensor.
		"""
		with torch.no_grad():
			if inputs.ndim == 2:
				inputs = torch.unsqueeze(inputs, 1)
				inputs = inputs.repeat(1, self.int_time_steps, 1)
			assert inputs.ndim == 3, \
				"shape of inputs must be (batch_size, time_steps, nb_features) or (batch_size, nb_features)"

			t_diff = self.int_time_steps - inputs.shape[1]
			assert t_diff >= 0, "inputs time steps must me less or equal to int_time_steps"
			if t_diff > 0:
				zero_inputs = torch.zeros(
					(inputs.shape[0], t_diff, inputs.shape[-1]),
					dtype=torch.float32,
					device=self.device
				)
				inputs = torch.cat([inputs, zero_inputs], dim=1)
		return inputs.float()

	def _format_hidden_outputs(
			self,
			hidden_states: Dict[str, List[Tuple[torch.Tensor, ...]]]
	) -> Dict[str, Tuple[torch.Tensor, ...]]:
		"""
		Permute the hidden states to have a dictionary of shape {layer_name: (tensor, ...)}
		:param hidden_states: Dictionary of hidden states
		:return: Dictionary of hidden states with the shape {layer_name: (tensor, ...)}
		"""
		hidden_states = {
			layer_name: tuple([torch.stack(e, dim=1) for e in list(zip(*trace))])
			for layer_name, trace in hidden_states.items()
		}
		return hidden_states

	def forward(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]],
			**kwargs
	) -> Tuple[Tensor, Dict[str, Tuple[Tensor, ...]]]:
		inputs = self.apply_transform(obs)
		inputs = {k: self._format_inputs(in_tensor) for k, in_tensor in inputs.items()}
		hidden_states = {
			layer_name: [None for t in range(self.int_time_steps + 1)]
			for layer_name in self.all_layer_names
		}
		outputs_trace: List[torch.Tensor] = []

		for t in range(1, self.int_time_steps + 1):
			features_list = []
			for layer_name, layer in self.input_layers.items():
				features, hidden_states[layer_name][t] = layer(inputs[layer_name][:, t - 1])
				features_list.append(features)
			if features_list:
				forward_tensor = torch.concat(features_list, dim=1)
			else:
				forward_tensor = torch.concat([inputs[in_name][:, t - 1] for in_name in inputs], dim=1)
			for layer_idx, (layer_name, layer) in enumerate(self.layers.items()):
				hh = hidden_states[layer_name][t - 1]
				forward_tensor, hidden_states[layer_name][t] = layer(forward_tensor, hh)
			outputs_trace.append(forward_tensor)

		hidden_states = {layer_name: trace[1:] for layer_name, trace in hidden_states.items()}
		hidden_states = self._format_hidden_outputs(hidden_states)
		outputs_trace_tensor = torch.stack(outputs_trace, dim=1)
		return outputs_trace_tensor, hidden_states

	def get_spikes_count_per_neuron(self, hidden_states: Dict[str, List[torch.Tensor]]) -> torch.Tensor:
		"""
		Get the spikes count per neuron from the hidden states
		:return:
		"""
		counts = []
		for l_name, traces in hidden_states.items():
			if isinstance(self.layers[l_name], LIFLayer):
				counts.extend(traces[-1].sum(dim=(0, 1)).tolist())
		return torch.tensor(counts, dtype=torch.float32, device=self.device)

	def get_logits(self, obs: Sequence[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
		output_records, hidden_states = self(obs)
		return output_records[:, -1]

	def _obs_forward_to_logits(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]]
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:param obs: shape: (observation_size, nb_agents, )
		:return:
		"""
		output_records, hidden_states = self(obs)
		output_values = output_records[:, -1]
		continuous_outputs = output_values[:, :self.spec.action_spec.continuous_size]
		discrete_outputs = []
		if self.spec.action_spec.discrete_size > 0:
			discrete_values = output_values[:, self.spec.action_spec.continuous_size:]
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
		:param as_numpy: True if the actions should be returned as numpy arrays.
		False if the actions should be returned as torch tensors, used for training.
		:param kwargs: Other arguments.
		:return: The actions.
		"""
		continuous_outputs, discrete_outputs = self._obs_forward_to_logits(obs)
		return TensorActionTuple(continuous_outputs, discrete_outputs)

	def get_actions_numpy(self, obs: Sequence[Union[np.ndarray, torch.Tensor]], **kwargs) -> ActionTuple:
		batch_size = obs[0].shape[0]
		assert all([batch_size == o.shape[0] for o in obs]), "All observations must have the same batch size"
		inputs = self.apply_transform(obs)
		output_records, hidden_states = self(inputs)
		output_values = output_records[:, -1].cpu().detach().numpy()
		actions = self.spec.action_spec.empty_action(output_values.shape[0])
		if self.spec.action_spec.continuous_size > 0:
			actions.add_continuous(output_values[:, :self.spec.action_spec.continuous_size])
		if self.spec.action_spec.discrete_size > 0:
			discrete_values = output_values[:, self.spec.action_spec.continuous_size:]
			discrete_action = np.zeros((batch_size, self.spec.action_spec.discrete_size))
			for branch_idx, branch in enumerate(self.spec.action_spec.discrete_branches):
				branch_cum_idx = sum(self.spec.action_spec.discrete_branches[:branch_idx])
				branch_values = discrete_values[:, branch_cum_idx:branch_cum_idx + branch]
				discrete_action[:, branch_idx] = np.argmax(branch_values, axis=-1)
			actions.add_discrete(discrete_action)
		return actions


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

	build_path = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe"
	integration_time = 100

	channel = EnvironmentParametersChannel()
	env = UnityEnvironment(file_name=build_path, seed=42, side_channels=[channel], no_graphics=True)
	channel.set_float_parameter("batchSize", 4)
	channel.set_float_parameter("camFollowTargetAgent", False)
	channel.set_float_parameter("droneMaxStartY", 1.1)
	channel.set_float_parameter("observationStacks", integration_time)
	channel.set_float_parameter("observationWidth", 28)
	channel.set_float_parameter("observationHeight", 28)
	env.reset()
	snn = SNNAgent(
		spec=env.behavior_specs[list(env.behavior_specs)[0]],
		behavior_name=list(env.behavior_specs)[0].split("?")[0],
		n_hidden_neurons=256,
		int_time_steps=integration_time,
		input_transform=[
			Compose([
				Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
				Lambda(lambda t: torch.permute(t, (2, 0, 1))),
				Lambda(lambda t: torch.flatten(t, start_dim=1))
			]),
			Compose([
				Lambda(lambda a: torch.from_numpy(a)),
			])
		]
	)
	hist = snn.fit(
		env,
		n_iterations=int(1e4),
		verbose=True,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		# force_overwrite=True,
	)
	# _, hist = snn.generate_trajectories(env, 1024, 0.0, verbose=True)
	# env.close()
	hist.plot(show=True, figsize=(10, 6))
