import time

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from torchvision.transforms import Compose, Lambda

from PythonAcademy.src.curriculum import CompletionCriteria, Curriculum, Lesson
from PythonAcademy.src.heuristics import FgHeuristic, RandomHeuristic
from PythonAcademy.src.snn_agent import LoadCheckpointMode, SNNAgent
from PythonAcademy.src.utils import to_tensor


def create_curriculum(channel, teacher=None):
	lessons = [
		Lesson(
			f"Alt{str(y).replace('.', '_')}",
			channel,
			params=dict(droneMaxStartY=y),
			# teacher=teacher if y <= 2.5 else None
		)
		for y in np.linspace(1.0, 10.0, num=100)
	]
	return Curriculum(lessons=lessons)


def setup_environment(integration_time):
	build_path = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe"

	params_channel = EnvironmentParametersChannel()
	engine_config_channel = EngineConfigurationChannel()
	env = UnityEnvironment(
		# file_name=build_path,
		seed=42,
		side_channels=[params_channel, engine_config_channel],
		no_graphics=True
	)
	engine_config_channel.set_configuration(EngineConfig.default_config())
	params_channel.set_float_parameter("batchSize", 4)
	params_channel.set_float_parameter("camFollowTargetAgent", False)
	params_channel.set_float_parameter("droneMaxStartY", 2.5)
	params_channel.set_float_parameter("observationStacks", integration_time)
	params_channel.set_float_parameter("observationWidth", 28)
	params_channel.set_float_parameter("observationHeight", 28)
	params_channel.set_float_parameter("enableNeuromorphicCamera", False)
	params_channel.set_float_parameter("divergenceAsOneHot", True)
	params_channel.set_float_parameter("enableDivergence", True)
	params_channel.set_float_parameter("usePositionAsInput", False)
	time.sleep(0.5)
	env.reset()
	time.sleep(0.5)
	return env, dict(params_channel=params_channel, engine_config_channel=engine_config_channel)


def train_agent(env, integration_time, channels):
	snn = SNNAgent(
		spec=env.behavior_specs[list(env.behavior_specs)[0]],
		behavior_name=list(env.behavior_specs)[0].split("?")[0],
		n_hidden_neurons=256,
		int_time_steps=integration_time,
		input_transform=[
			Compose([
				Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
				# Lambda(lambda t: torch.permute(t, (2, 0, 1))),
				# Lambda(lambda t: torch.flatten(t, start_dim=1))
			]),
			Compose([
				Lambda(lambda a: torch.from_numpy(a)),
			])
		]
	)
	hist = snn.fit(
		env,
		num_iterations=int(1e4),
		curriculum=create_curriculum(channels["params_channel"], teacher=FgHeuristic(env)),
		verbose=True,
		# load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		force_overwrite=True,
		curriculum_strength=1.0,
	)
	# _, hist = snn.generate_trajectories(env, 1024, 0.0, verbose=True)
	# env.close()
	hist.plot(show=True, figsize=(10, 6))


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	integration_time = 100
	env, channels = setup_environment(integration_time)
	train_agent(env, integration_time, channels)
	# for h in [FgHeuristic(env), RandomHeuristic(env)]:
	# 	print(f"{h.name} mean cumulative rewards: {np.mean([ex.reward for ex in h.buffer if ex.terminal]):.3f}")
	# lesson = Lesson(name="LowAltitude1DF", completion_criteria=0.9, teacher=h)
	try:
		env.close()
	except:
		pass




