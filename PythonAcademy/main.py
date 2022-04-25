import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from torchvision.transforms import Compose, Lambda

from PythonAcademy.src.curriculum import Lesson
from PythonAcademy.src.heuristic import Heuristic
from PythonAcademy.src.snn_agent import LoadCheckpointMode, SNNAgent
from PythonAcademy.src.utils import to_tensor


def setup_environment(integration_time):
	build_path = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe"

	channel = EnvironmentParametersChannel()
	env = UnityEnvironment(file_name=build_path, seed=42, side_channels=[channel], no_graphics=True)
	channel.set_float_parameter("batchSize", 4)
	channel.set_float_parameter("camFollowTargetAgent", False)
	channel.set_float_parameter("droneMaxStartY", 1.1)
	channel.set_float_parameter("observationStacks", integration_time)
	channel.set_float_parameter("observationWidth", 28)
	channel.set_float_parameter("observationHeight", 28)
	env.reset()
	return env


def train_agent(env, integration_time):
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


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	integration_time = 100
	env = setup_environment(integration_time)

	h = Heuristic(env)
	print(f"Heuristic mean cumulative rewards: {np.mean([ex.reward for ex in h.buffer]):.3f}")
	# lesson = Lesson(name="LowAltitude1DF", completion_criteria=0.9, teacher=h)




