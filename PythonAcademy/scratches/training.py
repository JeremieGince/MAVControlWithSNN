import time
from copy import deepcopy

import numpy as np
from mlagents_envs.environment import UnityEnvironment as UEnv
import torch

from PythonAcademy.models.dqn import DQN, ReplayBuffer, dqn_loss, format_batch
from PythonAcademy.models.fully_connected import SMNNModel
from PythonAcademy.models.short_memory_model import SMModel
from PythonAcademy.utils import show_rewards


def main(
		batch_size: int = 64,
		gamma: float = 0.99,
		buffer_size: int = int(1e5),
		seed: int = 42,
		tau: float = 1e-2,
		training_interval: int = 5,
		lr: float = 1e-3,
		epsilon_decay: float = 0.995,
		min_epsilon: float = 0.01,
		model_kwargs: dict = None,
		device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
		**kwargs
):
	if model_kwargs is None:
		model_kwargs = {}

	env = UEnv(file_name=None, seed=42, side_channels=[])
	env.reset()

	behavior_name = list(env.behavior_specs)[0]
	print(f"Name of the behavior: {behavior_name}")
	spec = env.behavior_specs[behavior_name]

	print(f"Number of observations: {len(spec.observation_specs)}")
	print(f"{spec.action_spec.continuous_size = }")

	actions = list(range(spec.action_spec.continuous_size))
	model: SMModel = kwargs.get("model_type", SMNNModel)(
		(len(spec.observation_specs), ),
		(spec.action_spec.continuous_size, ),
		memory_size=kwargs.get("memory_size", 1),
		**model_kwargs
	)
	policy_net = DQN(
		actions,
		model,
		optimizer=torch.optim.Adam(model.parameters(), lr=lr),
		loss_function=dqn_loss,
	)

	target_net = DQN(actions, deepcopy(model), optimizer="sgd", loss_function=dqn_loss, )
	replay_buffer = ReplayBuffer(buffer_size)

	model.to(device)
	policy_net.to(device)

	max_episode = kwargs.get("max_episode", 600)
	episodes_done = 0
	steps_done = 0
	epsilon = 1.0
	verbose_interval = kwargs.get("verbose_interval", 100)
	render_interval = kwargs.get("render_interval", verbose_interval)

	R_episodes = []
	start_time = time.time()
	best_score = -np.inf
	for episode in range(max_episode):
		env.reset()
		decision_steps, terminal_steps = env.get_steps(behavior_name)
		states = np.zeros((model.memory_size, len(spec.observation_specs)), len(decision_steps.agent_id))
		states[-1] = decision_steps.obs

		terminal = False
		R_episode: float = 0.0
		while not terminal:
			a = policy_net.get_action(state, epsilon)
			# next_frame, r, terminal, _ = environment.step(a)
			actions = spec.action_spec.random_action(len(decision_steps))
			env.set_actions(behavior_name, actions)
			env.step()
			decision_steps, terminal_steps = env.get_steps(behavior_name)

			next_state = np.vstack([np.delete(deepcopy(state), obj=0, axis=0), decision_steps.obs])

			replay_buffer.store((state, a, r, next_state, terminal))
			state = next_state
			steps_done += 1

			R_episode += r

			if steps_done % training_interval == 0:
				if len(replay_buffer.data) >= batch_size:
					batch = replay_buffer.get_random_batch(batch_size)
					x, y = format_batch(batch, target_net, gamma)
					loss = policy_net.train_on_batch(x, y)
					target_net.soft_update(policy_net, tau)

		R_episodes.append(R_episode)
		if episodes_done % verbose_interval == 0:
			if episodes_done == 0:
				print(f"episode: {episodes_done}, R: {R_episode:.2f}, epsilon: {epsilon:.2f}")
			else:
				print(
					f"episode: {episodes_done}, R: {R_episode:.2f}," 
					f" R_mean_100: {np.mean(R_episodes[-100:]):.2f}, epsilon: {epsilon:.2f}"
				)
		if episodes_done % render_interval == 0 and episodes_done > 0:
			show_rewards(
				R_episodes,
				block=False,
				title=kwargs.get("title", "RNN Rewards") + f", epi {episodes_done} " + f" env: {behavior_name}",
				subfolder=f"temp/{behavior_name}",
			)

		# if np.mean(R_episodes[-100:]) > best_score:
		#    best_score = np.mean(R_episodes[-100:])

		epsilon = max(min_epsilon, epsilon_decay * epsilon)
		episodes_done += 1

	show_rewards(
		R_episodes,
		block=True,
		title=kwargs.get("title", "RNN Rewards") + f" env: {behavior_name}",
		subfolder=f"{behavior_name}"
	)
	print(
		f"\n episodes: {episodes_done}," 
		f" R_mean_100: {np.mean(R_episodes[-100:]):.2f}," 
		f"Elapse time: {time.time() - start_time:.2f} [s] \n"
	)
	env.close()
	policy_net.save_weights(kwargs.get("filename_weights", "model_weights") + ".weights")


if __name__ == '__main__':
	# env = UEnv(file_name=None, seed=42, side_channels=[])
	# env.reset()
	#
	# behavior_name = list(env.behavior_specs)[0]
	# print(f"Name of the behavior: {behavior_name}")
	# spec = env.behavior_specs[behavior_name]
	#
	# print(f"Number of observations: {len(spec.observation_specs)}")
	# print(f"{spec.action_spec.continuous_size = }")
	#
	# for episode in range(10):
	# 	env.reset()
	# 	decision_steps, terminal_steps = env.get_steps(behavior_name)
	# 	tracked_agent = -1
	# 	done = False
	# 	episode_rewards = 0
	# 	while not done:
	# 		if tracked_agent == -1 and len(decision_steps) >= 1:
	# 			tracked_agent = decision_steps.agent_id[0]
	# 		print(f"{decision_steps.obs = }")
	# 		actions = spec.action_spec.random_action(len(decision_steps))
	# 		env.set_actions(behavior_name, actions)
	# 		env.step()
	# 		decision_steps, terminal_steps = env.get_steps(behavior_name)
	# 		if tracked_agent in decision_steps:  # The agent requested a decision
	# 			episode_rewards += decision_steps[tracked_agent].reward
	# 		if tracked_agent in terminal_steps:  # The agent terminated its episode
	# 			episode_rewards += terminal_steps[tracked_agent].reward
	# 			done = True
	# 	print(f"Total rewards for episode {episode} is {episode_rewards}")
	# env.close()
	main()
