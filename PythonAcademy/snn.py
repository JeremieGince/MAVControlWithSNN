from typing import Dict, List

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from torch import nn
from mlagents_envs.base_env import ActionTuple, BehaviorSpec, BaseEnv
import enum

from PythonAcademy.models.dqn import Experience, ReplayBuffer, Trajectory
from PythonAcademy.spike_funcs import HeavisideSigmoidApprox


class ReadoutMth(enum.Enum):
	RNN = 0


class ForwardMth(enum.Enum):
	LAYER_THEN_TIME = 0
	TIME_THEN_LAYER = 1


class SNN(torch.nn.Module):
	def __init__(
			self,
			spec: BehaviorSpec,
			n_hidden_neurons=None,
			use_recurrent_connection=True,
			dt=1e-3,
			tau_syn=10e-3,
			tau_mem=5e-3,
			spike_func=HeavisideSigmoidApprox.apply,
			device=None,
			forward_mth=ForwardMth.LAYER_THEN_TIME,
			readout_mth=ReadoutMth.RNN,
	):
		super(SNN, self).__init__()
		self.spec = spec
		self.device = device
		if self.device is None:
			self._set_default_device_()

		self.n_hidden_neurons = n_hidden_neurons if n_hidden_neurons is not None else []
		self.use_recurrent_connection = use_recurrent_connection
		self.forward_weights = nn.ParameterList()
		self.recurrent_weights = nn.ParameterList()
		self.readout_weights = nn.Parameter()
		self._populate_forward_weights_()
		self._populate_recurrent_weights_()
		self._populate_readout_weights_()
		self.initialize_weights_()

		self.dt = dt
		self.alpha = np.exp(-dt / tau_syn)
		self.beta = np.exp(-dt / tau_mem)
		self.spike_func = spike_func

		self.forward_func = self.get_forward_func(forward_mth)
		self.readout_func = self.get_readout_func(readout_mth)

		# TODO: split the inputs to two sets: spikes and currents. En utilisant le specs, on peut changer
		#  l'architecture du snn afin d'être en mesure de bien gérer les inputs. Les inputs de style onHot
		#  devraient être des entrée de formes spikes et les entrées de forment continue devraient être envoyer
		#  sous forme de courant. Idéalement les entrées sous forme d'image devraient être traité par un conv snn.
		#  Il devrait avoir autant de layer d'entrés que de sensor (len(self.spec.observation_specs)). Ces layers
		#  d'entrées devraient output sur une même hidden layer.

	@property
	def inputs_size(self):
		return sum([np.prod(entry.shape) for entry in self.spec.observation_specs])

	@property
	def output_size(self):
		return self.spec.action_spec.continuous_size + sum(self.spec.action_spec.discrete_branches)

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def _populate_forward_weights_(self):
		self.forward_weights = torch.nn.ParameterList()
		if not self.n_hidden_neurons:
			return
		self.forward_weights.append(
			nn.Parameter(
				torch.empty((self.inputs_size, self.n_hidden_neurons[0]), device=self.device),
				requires_grad=True
			)
		)
		for i, hn in enumerate(self.n_hidden_neurons[:-1]):
			self.forward_weights.append(
				nn.Parameter(torch.empty((hn, self.n_hidden_neurons[i + 1]), device=self.device), requires_grad=True)
			)

	def _populate_readout_weights_(self):
		if self.n_hidden_neurons:
			self.readout_weights = nn.Parameter(
				torch.empty((self.n_hidden_neurons[-1], self.output_size), device=self.device),
				requires_grad=True
			)
		else:
			self.readout_weights = nn.Parameter(
				torch.empty((self.inputs_size, self.output_size), device=self.device),
				requires_grad=True
			)

	def _populate_recurrent_weights_(self):
		self.recurrent_weights = nn.ParameterList()
		if not self.use_recurrent_connection:
			return
		for i, hn in enumerate(self.n_hidden_neurons):
			self.recurrent_weights.append(nn.Parameter(torch.empty((hn, hn), device=self.device), requires_grad=True))

	def initialize_weights_(self):
		for param in self.parameters():
			torch.nn.init.xavier_normal_(param)

	def get_readout_func(self, readout_mth: ReadoutMth):
		readout_mth_to_func = {
			ReadoutMth.RNN: self.forward_readout_rnn,
		}
		return readout_mth_to_func.get(readout_mth, ReadoutMth.RNN)

	def get_forward_func(self, forward_mth: ForwardMth):
		forward_mth_to_func = {
			ForwardMth.LAYER_THEN_TIME: self.forward_layer_time,
			ForwardMth.TIME_THEN_LAYER: self.forward_time_layer,
		}
		return forward_mth_to_func.get(forward_mth, ForwardMth.LAYER_THEN_TIME)

	def get_weights(self):
		return [*self.forward_weights, *self.recurrent_weights, self.readout_weights]

	def forward(self, inputs):
		spikes_records, membrane_potential_records = self.forward_func(inputs)
		output_records = self.readout_func(inputs, spikes_records, membrane_potential_records)
		return output_records, spikes_records, membrane_potential_records

	def forward_layer_time(self, inputs):
		membrane_potential_records = []
		spikes_records = [inputs, ]
		batch_size, nb_time_steps, nb_features = inputs.shape

		# Compute hidden layers activity
		for ell, f_weights in enumerate(self.forward_weights):
			h_ell = torch.einsum("btf, fo -> bto", (spikes_records[-1], f_weights))

			forward_current = torch.zeros((batch_size, f_weights.shape[-1]), device=self.device, dtype=torch.float)
			forward_potential = torch.zeros((batch_size, f_weights.shape[-1]), device=self.device, dtype=torch.float)
			spikes = torch.zeros((batch_size, f_weights.shape[-1]), device=self.device, dtype=torch.float)

			local_membrane_potential_records = []
			local_spikes_records = []

			for t in range(h_ell.shape[1]):
				if self.use_recurrent_connection:
					current_recurrent = torch.einsum("bo, of -> bf", (spikes, self.recurrent_weights[ell]))
				else:
					current_recurrent = 0.0

				spikes = self.spike_func(forward_potential - 1.0)
				is_active = 1.0 - spikes.detach()

				forward_current = self.alpha * forward_current + h_ell[:, t] + current_recurrent
				forward_potential = (self.beta * forward_potential + forward_current) * is_active

				local_membrane_potential_records.append(forward_potential)
				local_spikes_records.append(spikes)

			membrane_potential_records.append(torch.stack(local_membrane_potential_records, dim=1))
			spikes_records.append(torch.stack(local_spikes_records, dim=1))

		return spikes_records[1:], membrane_potential_records

	def forward_readout_rnn(self, inputs, spikes_records=None, membrane_potential_records=None):
		batch_size, nb_time_steps, nb_features = inputs.shape

		h_out = torch.einsum("btf, fo -> bto", (spikes_records[-1], self.readout_weights))
		forward_current = torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float)
		forward_potential = torch.zeros((batch_size, self.output_size), device=self.device, dtype=torch.float)
		output_records = [forward_potential, ]

		for t in range(h_out.shape[1]):
			forward_current = self.alpha * forward_current + h_out[:, t]
			forward_potential = (self.beta * forward_potential + forward_current)
			output_records.append(forward_potential)

		return torch.stack(output_records, dim=1)

	def forward_time_layer(self, inputs):
		membrane_potential_records = []
		spikes_records = []
		batch_size, input_size = inputs.shape

		forward_synaptic_currents = [
			torch.zeros((batch_size, f_weights.shape[0]), device=self.device, dtype=torch.float)
			for i, f_weights in enumerate(self.forward_weights)
		]
		forward_membrane_potentials = [
			torch.zeros((batch_size, f_weights.shape[0]), device=self.device, dtype=torch.float)
			for i, f_weights in enumerate(self.forward_weights)
		]
		spikes = [
			torch.zeros((batch_size, f_weights.shape[0]), device=self.device, dtype=torch.float)
			for i, f_weights in enumerate(self.forward_weights)
		]
		spikes[0] = inputs

		for t in range(self.int_time_steps):
			for ell, f_weights in enumerate(self.forward_weights):
				h_ell = torch.dot(spikes[ell], f_weights)
				forward_synaptic_currents[ell] = self.alpha * forward_membrane_potentials[ell] + h_ell
				forward_membrane_potentials[ell] = self.beta * forward_membrane_potentials[ell] + \
				                                   forward_synaptic_currents[ell]
				out = self.spike_func(forward_membrane_potentials[ell])
				spikes[ell] = out.detach()

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


if __name__ == '__main__':
	env = UnityEnvironment(file_name=None, seed=42, side_channels=[])
	env.reset()
	snn = SNN(spec=env.behavior_specs[list(env.behavior_specs)[0]])
	snn.generate_trajectories(env, 1024, 0.0)






