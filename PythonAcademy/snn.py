import json
import os
from typing import Any, Dict, Iterable, List, Tuple, Type, Union

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from torch import Tensor, nn
import torch.nn.functional as F
from mlagents_envs.base_env import ActionTuple, BehaviorSpec, BaseEnv
import enum

from PythonAcademy.models.dqn import Experience, ReplayBuffer, Trajectory
from PythonAcademy.spike_funcs import HeavisideSigmoidApprox, SpikeFuncType, SpikeFuncType2Func, SpikeFunction
from PythonAcademy.spiking_layers import LIFLayer, LayerType, LayerType2Layer, ReadoutLayer
from PythonAcademy.utils import LossHistory, mapping_update_recursively


class ReadoutMth(enum.Enum):
	RNN = 0


class ForwardMth(enum.Enum):
	LAYER_THEN_TIME = 0
	TIME_THEN_LAYER = 1


class LoadCheckpointMode(enum.Enum):
	BEST_EPOCH = enum.auto()
	LAST_EPOCH = enum.auto()


class SNN(torch.nn.Module):
	SAVE_EXT = '.pth'
	SUFFIX_SEP = '-'
	CHECKPOINTS_META_SUFFIX = 'checkpoints'
	CHECKPOINT_SAVE_PATH_KEY = "save_path"
	CHECKPOINT_BEST_KEY = "best"
	CHECKPOINT_EPOCHS_KEY = "epochs"
	CHECKPOINT_EPOCH_KEY = "epoch"
	CHECKPOINT_LOSS_KEY = 'loss'
	CHECKPOINT_OPTIMIZER_STATE_DICT_KEY = "optimizer_state_dict"
	CHECKPOINT_STATE_DICT_KEY = "model_state_dict"
	CHECKPOINT_FILE_STRUCT: Dict[str, Union[str, Dict[int, str]]] = {
		CHECKPOINT_BEST_KEY: CHECKPOINT_SAVE_PATH_KEY,
		CHECKPOINT_EPOCHS_KEY: {0: CHECKPOINT_SAVE_PATH_KEY},
	}
	load_mode_to_suffix = {mode: mode.name for mode in list(LoadCheckpointMode)}

	def __init__(
			self,
			spec: BehaviorSpec,
			n_hidden_neurons: Iterable[int] = None,
			use_recurrent_connection: Union[bool, Iterable[bool]] = True,
			int_time_steps=100,
			spike_func: Union[Type[SpikeFunction], SpikeFuncType] = HeavisideSigmoidApprox,
			hidden_layer_type: Union[Type[LIFLayer], LayerType] = LIFLayer,
			device=None,
			checkpoint_folder: str = "checkpoints",
			model_name: str = "snn",
			**kwargs
	):
		super(SNN, self).__init__()
		self.spec = spec
		self.kwargs = kwargs

		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.dt = self.kwargs.get("dt", 1e-3)
		self.int_time_steps = int_time_steps
		if isinstance(spike_func, SpikeFuncType):
			spike_func = SpikeFuncType2Func[spike_func]
		self.spike_func = spike_func
		if isinstance(hidden_layer_type, LayerType):
			hidden_layer_type = LayerType2Layer[hidden_layer_type]
		self.hidden_layer_type = hidden_layer_type

		self.checkpoint_folder = checkpoint_folder
		self.model_name = model_name

		if isinstance(n_hidden_neurons, int):
			n_hidden_neurons = [n_hidden_neurons]
		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else []
		self.use_recurrent_connection = use_recurrent_connection
		self.layers = nn.ModuleDict()
		self._add_layers_()
		self.initialize_weights_()
		self.loss_history = LossHistory()

		# TODO: split the inputs to two sets: spikes and currents. En utilisant le specs, on peut changer
		#  l'architecture du snn afin d'être en mesure de bien gérer les inputs. Les inputs de style onHot
		#  devraient être des entrée de formes spikes et les entrées de forment continue devraient être envoyer
		#  sous forme de courant. Idéalement les entrées sous forme d'image devraient être traité par un conv snn.
		#  Il devrait avoir autant de layer d'entrés que de sensor (len(self.spec.observation_specs)). Ces layers
		#  d'entrées devraient output sur une même hidden layer.

	@property
	def checkpoints_meta_path(self) -> str:
		return f"{self.checkpoint_folder}/{self.model_name}{SNN.SUFFIX_SEP}{SNN.CHECKPOINTS_META_SUFFIX}.json"

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def _add_input_layer_(self):
		if not self.n_hidden_neurons:
			return
		self.layers["input"] = self.hidden_layer_type(
			input_size=self.input_size,
			output_size=self.n_hidden_neurons[0],
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
			in_size = self.input_size
		self.layers["readout"] = ReadoutLayer(
			input_size=in_size,
			output_size=self.output_size,
			dt=self.dt,
			spike_func=self.spike_func,
			device=self.device,
			**self.kwargs
		)

	def _add_layers_(self):
		self._add_input_layer_()
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

	def forward(self, inputs):
		inputs = self._format_inputs(inputs)
		hidden_states = {
			layer_name: [None for t in range(self.int_time_steps + 1)]
			for layer_name, _ in self.layers.items()
		}
		outputs_trace: List[torch.Tensor] = []

		for t in range(1, self.int_time_steps + 1):
			forward_tensor = inputs[:, t - 1]
			for layer_idx, (layer_name, layer) in enumerate(self.layers.items()):
				hh = hidden_states[layer_name][t - 1]
				forward_tensor, hidden_states[layer_name][t] = layer(forward_tensor, hh)
			outputs_trace.append(forward_tensor)

		hidden_states = {layer_name: trace[1:] for layer_name, trace in hidden_states.items()}
		hidden_states = self._format_hidden_outputs(hidden_states)
		outputs_trace_tensor = torch.stack(outputs_trace, dim=1)
		return outputs_trace_tensor, hidden_states

	def get_prediction_logits(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		outputs_trace, hidden_states = self(inputs.to(self.device))
		logits, _ = torch.max(outputs_trace, dim=1)
		# logits = batchwise_temporal_filter(outputs_trace, decay=0.9)
		if re_outputs_trace and re_hidden_states:
			return logits, outputs_trace, hidden_states
		elif re_outputs_trace:
			return logits, outputs_trace
		elif re_hidden_states:
			return logits, hidden_states
		else:
			return logits

	def get_prediction_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		m, *outs = self.get_prediction_logits(inputs, re_outputs_trace, re_hidden_states)
		if re_outputs_trace or re_hidden_states:
			return F.softmax(m, dim=-1), *outs
		return m

	def get_prediction_log_proba(
			self,
			inputs: torch.Tensor,
			re_outputs_trace: bool = True,
			re_hidden_states: bool = True
	) -> Union[tuple[Tensor, Any, Any], tuple[Tensor, Any], Tensor]:
		m, *outs = self.get_prediction_logits(inputs, re_outputs_trace, re_hidden_states)
		if re_outputs_trace or re_hidden_states:
			return F.log_softmax(m, dim=-1), *outs
		return m

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

	def obs_to_inputs(self, obs):
		"""

		:param obs: shape: (observation_size, nb_agents, )
		:return:
		"""
		spikes = np.asarray(obs)  # TODO: add time dimension
		return torch.from_numpy(spikes)

	def get_action(self, inputs, epsilon: float = 0.0) -> ActionTuple:
		if np.random.random() < epsilon:
			return self.spec.action_spec.random_action(inputs.shape[0])
		output_records, spikes_records, membrane_potential_records = self.forward(inputs)
		output_values = output_records[-1].cpu().detach().numpy()
		action = self.spec.action_spec.empty_action(inputs.shape[0])
		if self.spec.action_spec.continuous_size > 0:
			action.add_continuous(output_values[:, :self.spec.action_spec.continuous_size])
		if self.spec.action_spec.discrete_size > 0:
			discrete_values = output_values[:, self.spec.action_spec.continuous_size:]
			discrete_action = np.zeros((inputs.shape[0], self.spec.action_spec.discrete_size))
			for branch_idx, branch in enumerate(self.spec.action_spec.discrete_branches):
				branch_cum_idx = sum(self.spec.action_spec.discrete_branches[:branch_idx])
				branch_values = discrete_values[:, branch_cum_idx:branch_cum_idx+branch]
				discrete_action[:, branch_idx] = np.argmax(branch_values, axis=-1)
			action.add_discrete(discrete_action)
		return action

	def fit(self, env: BaseEnv, buffer_size: int, epsilon: float):
		buffer = ReplayBuffer(buffer_size)
		env.reset()
		behavior_name = list(env.behavior_specs)[0]

	def generate_trajectories(self, env: BaseEnv, buffer_size: int, epsilon: float):
		# Create an empty Buffer
		buffer = ReplayBuffer(buffer_size)

		# Reset the environment
		env.reset()
		# Read and store the Behavior Name of the Environment
		behavior_name = list(env.behavior_specs)[0]

		# Create a Mapping from AgentId to Trajectories. This will help us create
		# trajectories for each Agents
		dict_trajectories_from_agent: Dict[int, Trajectory] = {}
		# Create a Mapping from AgentId to the last observation of the Agent
		dict_last_obs_from_agent: Dict[int, np.ndarray] = {}
		# Create a Mapping from AgentId to the last observation of the Agent
		dict_last_action_from_agent: Dict[int, np.ndarray] = {}
		# Create a Mapping from AgentId to cumulative reward (Only for reporting)
		dict_cumulative_reward_from_agent: Dict[int, float] = {}
		# Create a list to store the cumulative rewards obtained so far
		cumulative_rewards: List[float] = []

		while len(buffer) < buffer_size:  # While not enough data in the buffer
			# Get the Decision Steps and Terminal Steps of the Agents
			decision_steps, terminal_steps = env.get_steps(behavior_name)

			# For all Agents with a Terminal Step:
			for agent_id_terminated in terminal_steps:
				# Create its last experience (is last because the Agent terminated)
				last_experience = Experience(
					obs=dict_last_obs_from_agent[agent_id_terminated].copy(),
					reward=terminal_steps[agent_id_terminated].reward,
					done=not terminal_steps[agent_id_terminated].interrupted,
					action=dict_last_action_from_agent[agent_id_terminated].copy(),
					next_obs=terminal_steps[agent_id_terminated].obs[0],
				)
				# Clear its last observation and action (Since the trajectory is over)
				dict_last_obs_from_agent.pop(agent_id_terminated)
				dict_last_action_from_agent.pop(agent_id_terminated)
				# Report the cumulative reward
				cumulative_reward = (
						dict_cumulative_reward_from_agent.pop(agent_id_terminated)
						+ terminal_steps[agent_id_terminated].reward
				)
				cumulative_rewards.append(cumulative_reward)
				# Add the Trajectory and the last experience to the buffer
				buffer.extend(dict_trajectories_from_agent.pop(agent_id_terminated))
				buffer.store(last_experience)

			# For all Agents with a Decision Step:
			for agent_id_decisions in decision_steps:
				# If the Agent does not have a Trajectory, create an empty one
				if agent_id_decisions not in dict_trajectories_from_agent:
					dict_trajectories_from_agent[agent_id_decisions] = []
					dict_cumulative_reward_from_agent[agent_id_decisions] = 0

				# If the Agent requesting a decision has a "last observation"
				if agent_id_decisions in dict_last_obs_from_agent:
					# Create an Experience from the last observation and the Decision Step
					exp = Experience(
						obs=dict_last_obs_from_agent[agent_id_decisions].copy(),
						reward=decision_steps[agent_id_decisions].reward,
						done=False,
						action=dict_last_action_from_agent[agent_id_decisions].copy(),
						next_obs=decision_steps[agent_id_decisions].obs[0],
					)
					# Update the Trajectory of the Agent and its cumulative reward
					dict_trajectories_from_agent[agent_id_decisions].append(exp)
					dict_cumulative_reward_from_agent[agent_id_decisions] += decision_steps[agent_id_decisions].reward
				# Store the observation as the new "last observation"
				dict_last_obs_from_agent[agent_id_decisions] = decision_steps[agent_id_decisions].obs[0]

			env.set_actions(behavior_name, self.get_action(self.obs_to_inputs(decision_steps.obs), epsilon))
			# Perform a step in the simulation
			env.step()
		return buffer, np.mean(cumulative_rewards)

	def update_weights(self):
		raise NotImplementedError()

	def _create_checkpoint_path(self, epoch: int = -1):
		return f"./{self.checkpoint_folder}/{self.model_name}{SNN.SUFFIX_SEP}{SNN.CHECKPOINT_EPOCH_KEY}{epoch}{SNN.SAVE_EXT}"

	def _create_new_checkpoint_meta(self, epoch: int, best: bool = False) -> dict:
		save_path = self._create_checkpoint_path(epoch)
		new_info = {SNN.CHECKPOINT_EPOCHS_KEY: {epoch: save_path}}
		if best:
			new_info[SNN.CHECKPOINT_BEST_KEY] = save_path
		return new_info

	def save_checkpoint(
			self,
			optimizer,
			epoch: int,
			epoch_losses: Dict[str, Any],
			best: bool = False,
	):
		os.makedirs(self.checkpoint_folder, exist_ok=True)
		save_path = self._create_checkpoint_path(epoch)
		torch.save({
			SNN.CHECKPOINT_EPOCH_KEY: epoch,
			SNN.CHECKPOINT_STATE_DICT_KEY: self.state_dict(),
			SNN.CHECKPOINT_OPTIMIZER_STATE_DICT_KEY: optimizer.state_dict(),
			SNN.CHECKPOINT_LOSS_KEY: epoch_losses,
		}, save_path)
		self.save_checkpoints_meta(self._create_new_checkpoint_meta(epoch, best))

	@staticmethod
	def get_save_path_from_checkpoints(
			checkpoints_meta: Dict[str, Union[str, Dict[Any, str]]],
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
	) -> str:
		if load_checkpoint_mode == load_checkpoint_mode.BEST_EPOCH:
			return checkpoints_meta[SNN.CHECKPOINT_BEST_KEY]
		elif load_checkpoint_mode == load_checkpoint_mode.LAST_EPOCH:
			epochs_dict = checkpoints_meta[SNN.CHECKPOINT_EPOCHS_KEY]
			last_epoch: int = max([int(e) for e in epochs_dict])
			return checkpoints_meta[SNN.CHECKPOINT_EPOCHS_KEY][str(last_epoch)]
		else:
			raise ValueError()

	def get_checkpoints_loss_history(self) -> LossHistory:
		history = LossHistory()
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			meta: dict = json.load(jsonFile)
		checkpoints = [torch.load(path) for path in meta[SNN.CHECKPOINT_EPOCHS_KEY].values()]
		for checkpoint in checkpoints:
			history.concat(checkpoint[SNN.CHECKPOINT_LOSS_KEY])
		return history

	def load_checkpoint(
			self,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_EPOCH
	) -> dict:
		with open(self.checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		path = self.get_save_path_from_checkpoints(info, load_checkpoint_mode)
		checkpoint = torch.load(path)
		self.load_state_dict(checkpoint[SNN.CHECKPOINT_STATE_DICT_KEY], strict=True)
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


if __name__ == '__main__':
	env = UnityEnvironment(file_name=None, seed=42, side_channels=[])
	env.reset()
	snn = SNN(spec=env.behavior_specs[list(env.behavior_specs)[0]])
	snn.generate_trajectories(env, 1024, 0.0)






