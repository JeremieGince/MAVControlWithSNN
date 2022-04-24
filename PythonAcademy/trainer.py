import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from mlagents_envs.environment import ActionTuple, BaseEnv
from typing import Dict, List
import random

from PythonAcademy.models.dqn import DQN, Experience, ReplayBuffer, Trajectory


class Trainer:
	@staticmethod
	def generate_trajectories(
			env: BaseEnv, q_net: DQN, buffer_size: int, epsilon: float
	):
		"""
		Given a Unity Environment and a Q-Network, this method will generate a
		buffer of Experiences obtained by running the Environment with the Policy
		derived from the Q-Network.
		:param env: The UnityEnvironment used.
		:param q_net: The Q-Network used to collect the data.
		:param buffer_size: The minimum size of the buffer this method will return.
		:param epsilon: Will add a random normal variable with standard deviation.
		epsilon to the value heads of the Q-Network to encourage exploration.
		:returns: a Tuple containing the created buffer and the average cumulative
		the Agents obtained.
		"""
		# Create an empty Buffer
		buffer = ReplayBuffer(buffer_size)

		# Reset the environment
		env.reset()
		# Read and store the Behavior Name of the Environment
		behavior_name = list(env.behavior_specs)[0]
		# Read and store the Behavior Specs of the Environment
		spec = env.behavior_specs[behavior_name]

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

			# permute the tensor to go from NHWC to NCHW
			order = (0, 3, 1, 2)
			decision_steps.obs = [np.transpose(obs, order) for obs in decision_steps.obs]
			terminal_steps.obs = [np.transpose(obs, order) for obs in terminal_steps.obs]

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
					dict_cumulative_reward_from_agent[agent_id_decisions] += (
						decision_steps[agent_id_decisions].reward
					)
				# Store the observation as the new "last observation"
				dict_last_obs_from_agent[agent_id_decisions] = (
					decision_steps[agent_id_decisions].obs[0]
				)

			# Generate an action for all the Agents that requested a decision
			# Compute the values for each action given the observation
			actions_values = (
				q_net(torch.from_numpy(decision_steps.obs[0])).detach().numpy()
			)
			# Add some noise with epsilon to the values
			actions_values += epsilon * (
				np.random.randn(actions_values.shape[0], actions_values.shape[1])
			).astype(np.float32)
			# Pick the best action using argmax
			actions = np.argmax(actions_values, axis=1)
			actions.resize((len(decision_steps), 1))
			# Store the action that was picked, it will be put in the trajectory later
			for agent_index, agent_id in enumerate(decision_steps.agent_id):
				dict_last_action_from_agent[agent_id] = actions[agent_index]

			# Set the actions in the environment
			# Unity Environments expect ActionTuple instances.
			action_tuple = ActionTuple()
			action_tuple.add_discrete(actions)
			env.set_actions(behavior_name, action_tuple)
			# Perform a step in the simulation
			env.step()
		return buffer, np.mean(cumulative_rewards)

	@staticmethod
	def update_q_net(
			q_net: DQN,
			optimizer: torch.optim,
			buffer: ReplayBuffer,
			action_size: int
	):
		"""
		Performs an update of the Q-Network using the provided optimizer and buffer
		"""
		BATCH_SIZE = 1000
		NUM_EPOCH = 3
		GAMMA = 0.9
		batch_size = min(len(buffer), BATCH_SIZE)
		# random.shuffle(buffer)
		# Split the buffer into batches
		batches = [
			buffer[batch_size * start: batch_size * (start + 1)]
			for start in range(int(len(buffer) / batch_size))
		]
		for _ in range(NUM_EPOCH):
			for batch in batches:
				# Create the Tensors that will be fed in the network
				obs = torch.from_numpy(np.stack([ex.obs for ex in batch]))
				reward = torch.from_numpy(
					np.array([ex.reward for ex in batch], dtype=np.float32).reshape(-1, 1)
				)
				done = torch.from_numpy(
					np.array([ex.terminal for ex in batch], dtype=np.float32).reshape(-1, 1)
				)
				action = torch.from_numpy(np.stack([ex.action for ex in batch]))
				next_obs = torch.from_numpy(np.stack([ex.next_obs for ex in batch]))

				# Use the Bellman equation to update the Q-Network
				target = (
						reward
						+ (1.0 - done)
						* GAMMA
						* torch.max(q_net(next_obs).detach(), dim=1, keepdim=True).values
				)
				mask = torch.zeros((len(batch), action_size))
				mask.scatter_(1, action, 1)
				prediction = torch.sum(q_net(obs) * mask, dim=1, keepdim=True)
				criterion = torch.nn.MSELoss()
				loss = criterion(prediction, target)

				# Perform the backpropagation
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

	def run(self):
		pass


if __name__ == '__main__':
	from mlagents_envs.registry import default_registry
	# Create the GridWorld Environment from the registry
	env = default_registry["GridWorld"].make()
	print("GridWorld environment created.")
	num_actions = 5

	try:
		# Create a new Q-Network.
		qnet = VisualQNetwork((3, 64, 84), 126, num_actions)

		experiences = ReplayBuffer()
		optim = torch.optim.Adam(qnet.parameters(), lr=0.001)

		cumulative_rewards: List[float] = []

		# The number of training steps that will be performed
		NUM_TRAINING_STEPS = int(os.getenv('QLEARNING_NUM_TRAINING_STEPS', 70))
		# The number of experiences to collect per training step
		NUM_NEW_EXP = int(os.getenv('QLEARNING_NUM_NEW_EXP', 1000))
		# The maximum size of the Buffer
		BUFFER_SIZE = int(os.getenv('QLEARNING_BUFFER_SIZE', 10000))

		for n in range(NUM_TRAINING_STEPS):
			new_exp, _ = Trainer.generate_trajectories(env, qnet, NUM_NEW_EXP, epsilon=0.1)
			random.shuffle(experiences)
			if len(experiences) > BUFFER_SIZE:
				experiences = experiences[:BUFFER_SIZE]
			experiences.extend(new_exp)
			Trainer.update_q_net(qnet, optim, experiences, num_actions)
			_, rewards = Trainer.generate_trajectories(env, qnet, 100, epsilon=0)
			cumulative_rewards.append(rewards)
			print("Training step ", n + 1, "\treward ", rewards)
	except KeyboardInterrupt:
		print("\nTraining interrupted, continue to next cell to save to save the model.")
	finally:
		env.close()

	# Show the training graph
	try:
		plt.plot(range(NUM_TRAINING_STEPS), cumulative_rewards)
	except ValueError:
		print("\nPlot failed on interrupted training.")
