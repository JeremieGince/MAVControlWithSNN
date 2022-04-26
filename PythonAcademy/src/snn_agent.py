import json
import os
import shutil
import warnings
from copy import deepcopy
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Type, Union
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from mlagents_envs.environment import UnityEnvironment
from torch import Tensor, nn
import torch.nn.functional as F
from mlagents_envs.base_env import ActionTuple, BehaviorSpec, BaseEnv, DecisionSteps, DimensionProperty, TerminalSteps
import enum

from torchvision.transforms import ToTensor, Compose, Lambda

from PythonAcademy.src.buffers import BatchExperience, Experience, ReplayBuffer, Trajectory
from PythonAcademy.src.curriculum import Curriculum
from PythonAcademy.src.spike_funcs import HeavisideSigmoidApprox, SpikeFuncType, SpikeFuncType2Func, SpikeFunction
from PythonAcademy.src.spiking_layers import ALIFLayer, LIFLayer, LayerType, LayerType2Layer, ReadoutLayer
from PythonAcademy.src.utils import TrainingHistory, linear_decay, mapping_update_recursively, to_tensor


class AgentsHistoryMaps:
	"""
	Class to store the mapping between agents and their history maps

	Attributes:
		trajectories (Dict[int, Trajectory]): Mapping between agent ids and their trajectories
		last_obs (Dict[int, np.ndarray]): Mapping between agent ids and their last observations
		last_action (Dict[int, np.ndarray]): Mapping between agent ids and their last actions
		cumulative_reward (Dict[int, float]): Mapping between agent ids and their cumulative rewards
	"""

	def __init__(self):
		self.trajectories: Dict[int, Trajectory] = defaultdict(list)
		self.last_obs: Dict[int, Any] = defaultdict()
		self.last_action: Dict[int, ActionTuple] = defaultdict()
		self.cumulative_reward: Dict[int, float] = defaultdict(lambda: 0.0)


class LoadCheckpointMode(enum.Enum):
	BEST_ITR = enum.auto()
	LAST_ITR = enum.auto()


class SNNAgent(torch.nn.Module):
	SAVE_EXT = '.pth'
	SUFFIX_SEP = '-'
	CHECKPOINTS_META_SUFFIX = 'checkpoints'
	CHECKPOINT_SAVE_PATH_KEY = "save_path"
	CHECKPOINT_BEST_KEY = "best"
	CHECKPOINT_ITRS_KEY = "iterations"
	CHECKPOINT_ITR_KEY = "itr"
	CHECKPOINT_METRICS_KEY = 'rewards'
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
	CHECKPOINT_STATE_DICT_KEY = "model_state_dict"
	CHECKPOINT_TRAINING_HISTORY_KEY = "training_history"
	CHECKPOINT_FILE_STRUCT: Dict[str, Union[str, Dict[int, str]]] = {
		CHECKPOINT_BEST_KEY: CHECKPOINT_SAVE_PATH_KEY,
		CHECKPOINT_ITRS_KEY: {0: CHECKPOINT_SAVE_PATH_KEY},
	}
	load_mode_to_suffix = {mode: mode.name for mode in list(LoadCheckpointMode)}

	def __init__(
			self,
			spec: BehaviorSpec,
			behavior_name: str,
			n_hidden_neurons: Union[Iterable[int], int] = None,
			use_recurrent_connection: Union[bool, Iterable[bool]] = True,
			int_time_steps: int = None,
			spike_func: Union[Type[SpikeFunction], SpikeFuncType] = HeavisideSigmoidApprox,
			hidden_layer_type: Union[Type[LIFLayer], LayerType] = ALIFLayer,
			device=None,
			checkpoint_folder: str = "checkpoints",
			model_name: str = "snn",
			input_transform: Union[Dict[str, Callable], List[Callable]] = None,
			**kwargs
	):
		super(SNNAgent, self).__init__()
		self.spec = spec
		self.kwargs = kwargs

		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.dt = self.kwargs.get("dt", 1e-3)
		self.int_time_steps = self._get_and_check_int_time_steps() if int_time_steps is None else int_time_steps
		if isinstance(spike_func, SpikeFuncType):
			spike_func = SpikeFuncType2Func[spike_func]
		self.spike_func = spike_func
		if isinstance(hidden_layer_type, LayerType):
			hidden_layer_type = LayerType2Layer[hidden_layer_type]
		self.hidden_layer_type = hidden_layer_type

		self.checkpoint_folder = checkpoint_folder
		self.model_name = model_name
		self.behavior_name = behavior_name

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
		self.training_history = TrainingHistory()

		if input_transform is None:
			input_transform = self.get_default_transform()
		if isinstance(input_transform, list):
			input_transform = {in_name: t for in_name, t in zip(self.input_sizes, input_transform)}
		self.input_transform: Dict[str, Callable] = input_transform
		self._add_to_device_transform_()

		self.continuous_criterion = nn.MSELoss()
		self.discrete_criterion = nn.MSELoss()

	@property
	def checkpoints_meta_path(self) -> str:
		full_filename = (
			f"{self.model_name}_{self.behavior_name}{SNNAgent.SUFFIX_SEP}{SNNAgent.CHECKPOINTS_META_SUFFIX}"
		)
		return f"{self.checkpoint_folder}/{full_filename}.json"

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

	def get_default_transform(self) -> Dict[str, nn.Module]:
		return {
			in_name: Compose([
				Lambda(lambda a: torch.from_numpy(a)),
			])
			for in_name in self.input_sizes
		}

	def _add_to_device_transform_(self):
		for in_name, trans in self.input_transform.items():
			self.input_transform[in_name] = Compose([trans, Lambda(lambda t: t.to(self.device))])

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

	def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[Tensor, Dict[str, Tuple[Tensor, ...]]]:
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

	def obs_to_inputs(self, obs: Sequence[Union[np.ndarray, torch.Tensor]]) -> Dict[str, torch.Tensor]:
		"""
		:param obs: shape: (observation_size, nb_agents, )
		:return:
		"""
		inputs = {
			obs_spec.name: torch.stack(
				[self.input_transform[obs_spec.name](obs_i) for obs_i in obs[obs_index]],
				dim=0
			)
			for obs_index, obs_spec in enumerate(self.spec.observation_specs)
		}
		return inputs

	def _obs_forward_to_logits(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]]
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		:param obs: shape: (observation_size, nb_agents, )
		:return:
		"""
		inputs = self.obs_to_inputs(obs)
		output_records, hidden_states = self(inputs)
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

	def get_actions(self, inputs: Dict[str, torch.Tensor], epsilon: float = 0.0) -> ActionTuple:
		batch_size = inputs[self.spec.observation_specs[0].name].shape[0]
		assert all([inputs[obs_spec.name].shape[0] == batch_size for obs_spec in self.spec.observation_specs])
		if np.random.random() < epsilon:
			return self.spec.action_spec.random_action(batch_size)
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

	@staticmethod
	def unbatch_actions(actions: ActionTuple) -> List[ActionTuple]:
		"""
		:param actions: shape: (batch_size, ...)
		:return:
		"""
		actions_list = []
		continuous, discrete = actions.continuous, actions.discrete
		batch_size = actions.continuous.shape[0] if continuous is not None else discrete.shape[0]
		assert batch_size is not None
		for i in range(batch_size):
			actions_list.append(
				ActionTuple(
					continuous[i] if continuous is not None else None,
					discrete[i] if discrete is not None else None)
			)
		return actions_list

	def _init_target_network(self) -> 'SNNAgent':
		target_network = deepcopy(self)
		target_network.eval()
		for param in target_network.parameters():
			param.requires_grad = False
		target_network.train()
		return target_network

	@staticmethod
	def _set_default_fit_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
		kwargs.setdefault("close_env", True)
		kwargs.setdefault("n_epochs", 3)
		kwargs.setdefault("init_lr", 1e-3)
		kwargs.setdefault("min_lr", 3.0e-4)
		kwargs.setdefault("weight_decay", 1e-5)
		kwargs.setdefault("init_epsilon", 0.99)
		kwargs.setdefault("epsilon_decay", 0.999)
		kwargs.setdefault("min_epsilon", 0.01)
		kwargs.setdefault("gamma", 0.99)
		kwargs.setdefault("tau", 0.01)
		kwargs.setdefault("n_batches", 3)
		kwargs.setdefault("update_freq", 5)
		kwargs.setdefault("curriculum_strength", 0.5)
		return kwargs

	def fit(
			self,
			env: BaseEnv,
			num_iterations: int = int(1e6),
			buffer_size: int = int(2**14),
			batch_size: int = 256,
			curriculum: Curriculum = None,
			load_checkpoint_mode: LoadCheckpointMode = None,
			force_overwrite: bool = False,
			save_freq: int = int(1e2),
			verbose: bool = True,
			**kwargs
	) -> TrainingHistory:
		# TODO: add learning rate decay
		kwargs = self._set_default_fit_kwargs(kwargs)
		assert batch_size <= buffer_size
		assert batch_size > 0
		optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["init_lr"], weight_decay=kwargs["weight_decay"])

		start_itr = 0
		if load_checkpoint_mode is None:
			if os.path.exists(self.checkpoints_meta_path):
				if force_overwrite:
					shutil.rmtree(self.checkpoint_folder)
				else:
					raise ValueError(
						f"{self.checkpoints_meta_path} already exists. "
						f"Set force_overwrite flag to True to overwrite existing saves."
					)
		else:
			try:
				checkpoint = self.load_checkpoint(load_checkpoint_mode)
				self.load_state_dict(checkpoint[SNNAgent.CHECKPOINT_STATE_DICT_KEY], strict=True)
				optimizer.load_state_dict(checkpoint[SNNAgent.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY])
				start_itr = int(checkpoint[SNNAgent.CHECKPOINT_ITR_KEY]) + 1
				# self.training_history = self.get_checkpoints_training_history()
				self.training_history: TrainingHistory = checkpoint[SNNAgent.CHECKPOINT_TRAINING_HISTORY_KEY]
			except FileNotFoundError:
				if verbose:
					warnings.warn("No such checkpoint. Fit from beginning.")

		target_network = self._init_target_network()
		best_rewards = self.training_history.max("Rewards")
		env.reset()
		buffer, _ = self.generate_trajectories(
			env, batch_size, epsilon=kwargs["init_epsilon"], verbose=verbose, **kwargs
		)
		p_bar = tqdm(range(start_itr, num_iterations), disable=not verbose, desc="Training")
		for i in p_bar:
			epsilon = linear_decay(kwargs["init_epsilon"], kwargs["min_epsilon"], kwargs["epsilon_decay"], i)
			teacher_loss = self.fit_curriculum_buffer(curriculum, optimizer, batch_size, **kwargs)
			itr_metrics = self._exec_fit_itr_(target_network, optimizer, env, epsilon, buffer, batch_size, **kwargs)
			p_bar_postfix = dict(
				loss=f"{itr_metrics['Loss']:.3f}",
				cum_rewards=f"{itr_metrics['Rewards']:.3f}",
				best_rewards=f"{best_rewards:.3f}",
			)
			if teacher_loss is not None:
				itr_metrics.update(TeacherLoss=teacher_loss)
				p_bar_postfix.update(TeacherLoss=f"{teacher_loss:.3f}")
			self.training_history.concat(itr_metrics)
			if curriculum is not None:
				msg = curriculum.on_iteration_end(self.training_history)
				p_bar_postfix.update(msg)
			p_bar.set_postfix(p_bar_postfix)
			if i % save_freq == 0:
				is_best = itr_metrics["Rewards"] > best_rewards
				if is_best:
					best_rewards = itr_metrics["Rewards"]
				self.save_checkpoint(i, itr_metrics, target_network, optimizer, is_best)
				self.plot_training_history(show=False)
		p_bar.close()
		if kwargs.get("close_env", True):
			env.close()
		self.hard_update(target_network)
		self.plot_training_history(show=False)
		return self.training_history

	def _exec_fit_itr_(
			self,
			target_network: 'SNNAgent',
			optimizer: torch.optim.Optimizer,
			env: BaseEnv,
			epsilon: float,
			buffer: ReplayBuffer,
			batch_size: int,
			**kwargs
	) -> Dict[str, float]:
		buffer, cumulative_rewards = self.generate_trajectories(
			env, kwargs["update_freq"], buffer, epsilon, verbose=False, p_bar_position=0, **kwargs
		)
		cum_rewards = np.mean(cumulative_rewards)
		itr_loss = self.fit_buffer(buffer, target_network, optimizer, batch_size, **kwargs)
		return dict(Rewards=cum_rewards, Loss=itr_loss)

	def fit_curriculum_buffer(
			self,
			curriculum: Curriculum,
			optimizer: torch.optim,
			batch_size: int,
			**kwargs
	) -> Optional[float]:
		"""
		Fit the curriculum buffer.
		:param curriculum:
		:param optimizer:
		:param batch_size:
		:param kwargs:
		:return:
		"""
		kwargs = self._set_default_fit_kwargs(kwargs)
		buffer = curriculum.teacher_buffer
		if buffer is None:
			return None
		batch_size = min(len(buffer), batch_size)
		batches = buffer.get_batch_generator(batch_size, kwargs["n_batches"], randomize=True, device=self.device)
		losses = []
		for _ in range(kwargs["n_epochs"]):
			for batch in batches:
				predictions = self._obs_forward_to_logits(batch.obs)
				loss = self._behaviour_cloning_update_weights(batch, predictions, optimizer, **kwargs)
				losses.append(loss)
		return float(np.mean(losses))

	def fit_buffer(
			self,
			buffer: ReplayBuffer,
			target_network: 'SNNAgent',
			optimizer: torch.optim,
			batch_size: int = 256,
			**kwargs
	) -> float:
		kwargs = self._set_default_fit_kwargs(kwargs)
		"""
		Performs an update of the Q-Network using the provided optimizer and buffer
		"""
		batch_size = min(len(buffer), batch_size)
		batches = buffer.get_batch_generator(batch_size, kwargs["n_batches"], randomize=True, device=self.device)
		losses = []
		for _ in range(kwargs["n_epochs"]):
			for batch in batches:
				predictions = self._obs_forward_to_logits(batch.obs)
				targets = target_network._obs_forward_to_logits(batch.next_obs)
				loss = self.update_weights(batch, predictions, targets, optimizer, kwargs["gamma"])
				target_network.soft_update(self, tau=kwargs["tau"])
				losses.append(loss)
		return float(np.mean(losses))

	def _behaviour_cloning_compute_continuous_loss(self, batch: BatchExperience, predictions, targets) -> torch.Tensor:
		if torch.numel(batch.continuous_actions) == 0:
			continuous_loss = 0.0
		else:
			continuous_loss = self.continuous_criterion(predictions, targets.to(self.device))
		return continuous_loss

	def _behaviour_cloning_compute_discrete_loss(self, batch: BatchExperience, predictions, targets) -> torch.Tensor:
		if torch.numel(batch.discrete_actions) == 0:
			discrete_loss = 0.0
		else:
			warnings.warn("Discrete loss is not implemented with cross entropy loss. This is a temporary solution.")
			discrete_loss = self.discrete_criterion(predictions, targets.to(self.device))
		return discrete_loss

	def _behaviour_cloning_update_weights(
			self,
			batch: BatchExperience,
			predictions: Tuple[torch.Tensor, torch.Tensor],
			optimizer: torch.optim,
			**kwargs
	) -> float:
		kwargs = self._set_default_fit_kwargs(kwargs)
		assert torch.numel(batch.continuous_actions) + torch.numel(batch.discrete_actions) > 0
		continuous_loss = self._behaviour_cloning_compute_continuous_loss(
			batch, predictions[0], batch.continuous_actions
		)
		discrete_loss = self._behaviour_cloning_compute_discrete_loss(
			batch, predictions[1], batch.discrete_actions
		)
		loss = (continuous_loss + discrete_loss) * kwargs["curriculum_strength"]
		# Perform the backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		return loss.detach().cpu().numpy().item()

	def _compute_continuous_loss(self, batch: BatchExperience, predictions, targets, gamma: float) -> torch.Tensor:
		if torch.numel(batch.continuous_actions) == 0:
			continuous_loss = 0.0
		else:
			targets = (
					batch.rewards
					+ (1.0 - batch.terminals)
					* gamma
					* targets
			).to(self.device)
			continuous_loss = self.continuous_criterion(predictions, targets)
		return continuous_loss

	def _compute_discrete_loss(self, batch: BatchExperience, predictions, targets, gamma: float) -> torch.Tensor:
		if torch.numel(batch.discrete_actions) == 0:
			discrete_loss = 0.0
		else:
			targets = (
					batch.rewards
					+ (1.0 - batch.terminals)
					* gamma
					* targets
			).to(self.device)
			warnings.warn("Discrete loss is not implemented with cross entropy loss. This is a temporary solution.")
			discrete_loss = self.discrete_criterion(predictions, targets)
		return discrete_loss

	def update_weights(
			self,
			batch: BatchExperience,
			predictions,
			targets,
			optimizer,
			gamma: float
	) -> float:
		"""
		Performs a single update of the Q-Network using the provided optimizer and buffer
		"""
		assert torch.numel(batch.continuous_actions) + torch.numel(batch.discrete_actions) > 0
		continuous_loss = self._compute_continuous_loss(batch, predictions[0], targets[0], gamma)
		discrete_loss = self._compute_discrete_loss(batch, predictions[1], targets[1], gamma)
		loss = continuous_loss + discrete_loss
		# Perform the backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		return loss.detach().cpu().numpy().item()
	
	def soft_update(self, other: 'SNNAgent', tau: float = 1e-2) -> None:
		"""
		Copies the weights from the other network to this network with a factor of tau
		"""
		with torch.no_grad():
			for param, other_param in zip(self.parameters(), other.parameters()):
				param.data.copy_((1 - tau) * param.data + tau * other_param.data)
	
	def hard_update(self, other: 'SNNAgent') -> None:
		"""
		Copies the weights from the other network to this network
		"""
		with torch.no_grad():
			self.load_state_dict(other.state_dict())

	@staticmethod
	def exec_terminal_steps_(
			agent_maps: AgentsHistoryMaps,
			buffer: ReplayBuffer,
			terminal_steps: TerminalSteps
	) -> List[float]:
		"""
		Execute terminal steps and return the rewards
		:param agent_maps: The agent maps
		:param buffer: The replay buffer
		:param terminal_steps: The terminal steps
		:return: The rewards
		"""
		cumulative_rewards = []
		# For all Agents with a Terminal Step:
		for agent_id_terminated in terminal_steps:
			# Create its last experience (is last because the Agent terminated)
			last_experience = Experience(
				obs=deepcopy(agent_maps.last_obs[agent_id_terminated]),
				reward=terminal_steps[agent_id_terminated].reward,
				terminal=not terminal_steps[agent_id_terminated].interrupted,
				action=deepcopy(agent_maps.last_action[agent_id_terminated]),
				next_obs=terminal_steps[agent_id_terminated].obs,
			)
			agent_maps.trajectories[agent_id_terminated].append(last_experience)
			# Clear its last observation and action (Since the trajectory is over)
			agent_maps.last_obs.pop(agent_id_terminated)
			agent_maps.last_action.pop(agent_id_terminated)
			# Report the cumulative reward
			cumulative_rewards.append(
				agent_maps.cumulative_reward.pop(agent_id_terminated, 0.0)
				+ terminal_steps[agent_id_terminated].reward
			)
			# Add the Trajectory to the buffer
			buffer.extend(agent_maps.trajectories.pop(agent_id_terminated))
			buffer.increment_counter()
		return cumulative_rewards

	@staticmethod
	def exec_decisions_steps_(agent_maps: AgentsHistoryMaps, decision_steps: DecisionSteps):
		"""
		Execute the decision steps of the agents
		:param agent_maps: The AgentMaps
		:param decision_steps: The decision steps
		:return: None
		"""
		# For all Agents with a Decision Step:
		for agent_id_decisions in decision_steps:
			# If the Agent requesting a decision has a "last observation"
			if agent_id_decisions in agent_maps.last_obs:
				# Create an Experience from the last observation and the Decision Step
				exp = Experience(
					obs=deepcopy(agent_maps.last_obs[agent_id_decisions]),
					reward=decision_steps[agent_id_decisions].reward,
					terminal=False,
					action=deepcopy(agent_maps.last_action[agent_id_decisions]),
					next_obs=decision_steps[agent_id_decisions].obs,
				)
				# Update the Trajectory of the Agent and its cumulative reward
				agent_maps.trajectories[agent_id_decisions].append(exp)
				agent_maps.cumulative_reward[agent_id_decisions] += decision_steps[agent_id_decisions].reward
			# Store the observation as the new "last observation"
			agent_maps.last_obs[agent_id_decisions] = decision_steps[agent_id_decisions].obs

	def generate_trajectories(
			self,
			env: BaseEnv,
			n_trajectories: int,
			buffer: Optional[ReplayBuffer] = None,
			epsilon: float = 0.0,
			verbose: bool = False,
			p_bar_position: int = 0,
			**kwargs
	) -> Tuple[ReplayBuffer, List[float]]:
		kwargs = self._set_default_fit_kwargs(kwargs)
		if buffer is None:
			buffer = ReplayBuffer(n_trajectories)
		env.reset()
		behavior_name = list(env.behavior_specs)[0]
		agent_maps = AgentsHistoryMaps()
		cumulative_rewards: List[float] = []
		p_bar = tqdm(
			total=n_trajectories, disable=not verbose, desc="Generating Trajectories", position=p_bar_position
		)
		last_buffer_counter = 0
		buffer.reset_counter()
		while buffer.counter < n_trajectories:  # While not enough data in the buffer
			decision_steps, terminal_steps = env.get_steps(behavior_name)
			cumulative_rewards.extend(self.exec_terminal_steps_(agent_maps, buffer, terminal_steps))
			if len(decision_steps) > 0:
				self.exec_decisions_steps_(agent_maps, decision_steps)
				actions = self.get_actions(self.obs_to_inputs(decision_steps.obs), epsilon)
				actions_list = self.unbatch_actions(actions)
				for agent_index, agent_id in enumerate(decision_steps.agent_id):
					agent_maps.last_action[agent_id] = actions_list[agent_index]
				env.set_actions(behavior_name, actions)
			p_bar.set_postfix(cumulative_reward=f"{np.mean(cumulative_rewards) if cumulative_rewards else 0.0:.3f}")
			p_bar.update(min(buffer.counter - last_buffer_counter, n_trajectories - last_buffer_counter))
			last_buffer_counter = buffer.counter
			env.step()
		p_bar.close()
		return buffer, cumulative_rewards

	def _create_checkpoint_path(self, epoch: int = -1):
		return f"./{self.checkpoint_folder}/{self.model_name}{SNNAgent.SUFFIX_SEP}{SNNAgent.CHECKPOINT_ITR_KEY}{epoch}{SNNAgent.SAVE_EXT}"

	def _create_new_checkpoint_meta(self, itr: int, best: bool = False) -> dict:
		save_path = self._create_checkpoint_path(itr)
		new_info = {SNNAgent.CHECKPOINT_ITRS_KEY: {itr: save_path}}
		if best:
			new_info[SNNAgent.CHECKPOINT_BEST_KEY] = save_path
		return new_info

	def save_checkpoint(
			self,
			itr: int,
			itr_metrics: Dict[str, Any],
			network=None,
			optimizer=None,
			best: bool = False,
	):
		if network is None:
			network = self
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		save_path = self._create_checkpoint_path(itr)
		torch.save({
			SNNAgent.CHECKPOINT_ITR_KEY                 : itr,
			SNNAgent.CHECKPOINT_STATE_DICT_KEY          : network.state_dict(),
			SNNAgent.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict() if optimizer is not None else None,
			SNNAgent.CHECKPOINT_METRICS_KEY             : itr_metrics,
			SNNAgent.CHECKPOINT_TRAINING_HISTORY_KEY    : self.training_history,
		}, save_path)
		self.save_checkpoints_meta(self._create_new_checkpoint_meta(itr, best))

	@staticmethod
	def get_save_path_from_checkpoints(
			checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> str:
		if load_checkpoint_mode == load_checkpoint_mode.BEST_ITR:
			return checkpoints_meta[SNNAgent.CHECKPOINT_BEST_KEY]
		elif load_checkpoint_mode == load_checkpoint_mode.LAST_ITR:
			itr_dict = checkpoints_meta[SNNAgent.CHECKPOINT_ITRS_KEY]
			last_itr: int = max([int(e) for e in itr_dict])
			return checkpoints_meta[SNNAgent.CHECKPOINT_ITRS_KEY][str(last_itr)]
		else:
			raise ValueError()

	def get_checkpoints_training_history(self) -> TrainingHistory:
		history = TrainingHistory()
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			meta: dict = json.load(jsonFile)
		checkpoints = [torch.load(path) for path in meta[SNNAgent.CHECKPOINT_ITRS_KEY].values()]
		for checkpoint in checkpoints:
			history.concat(checkpoint[SNNAgent.CHECKPOINT_METRICS_KEY])
		return history

	def load_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> dict:
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		path = self.get_save_path_from_checkpoints(info, load_checkpoint_mode)
		checkpoint = torch.load(path)
		self.load_state_dict(checkpoint[SNNAgent.CHECKPOINT_STATE_DICT_KEY], strict=True)
		return checkpoint

	def to_onnx(self, in_viz=None):
		if in_viz is None:
			in_viz = torch.randn((1, self.input_size), device=self.device)
		torch.onnx.export(
			self,
			in_viz,
			f"{self.checkpoint_folder}/{self.model_name}.onnx",
			verbose=True,
			input_names=None,
			output_names=None,
			opset_version=11
		)

	def save_checkpoints_meta(self, new_info: dict):
		info = dict()
		if os.path.exists(self.checkpoints_meta_path):
			with open(self.checkpoints_meta_path, "r+") as jsonFile:
				info = json.load(jsonFile)
		mapping_update_recursively(info, new_info)
		with open(self.checkpoints_meta_path, "w+") as jsonFile:
			json.dump(info, jsonFile, indent=4)

	def plot_training_history(self, training_history: TrainingHistory = None, show: bool = False) -> str:
		if training_history is None:
			training_history = self.training_history
		save_path = f"./{self.checkpoint_folder}/training_history.png"
		os.makedirs(f"./{self.checkpoint_folder}/", exist_ok=True)
		training_history.plot(save_path, show)
		return save_path


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
	from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig

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
		num_iterations=int(1e4),
		verbose=True,
		load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		# force_overwrite=True,
	)
	# _, hist = snn.generate_trajectories(env, 1024, 0.0, verbose=True)
	# env.close()
	hist.plot(show=True, figsize=(10, 6))
