import time
from typing import Any, Dict

import numpy as np
import torch
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from torchvision.transforms import Compose, Lambda

from PythonAcademy.src.curriculum import CompletionCriteria, Curriculum, Lesson
from PythonAcademy.src.heuristics import FgHeuristic
from PythonAcademy.src.mlp_agent import MLPAgent
from PythonAcademy.src.rl_academy import LoadCheckpointMode, RLAcademy
from PythonAcademy.src.utils import send_parameter_to_channel, threshold_image, to_tensor


def get_env_parameters(int_time: int):
	return dict(
		n_agents=8,
		camFollowTargetAgent=False,
		droneMaxStartY=2.5,
		observationStacks=int_time,
		observationWidth=28,
		observationHeight=28,
		enableNeuromorphicCamera=False,
		enableCamera=False,
		usePositionAsInput=False,
		enableTorque=False,
		enableDisplacement=False,
		useRotationAsInput=False,
		useVelocityAsInput=False,
		useAngularVelocityAsInput=False,
		useDivergenceAsInput=True,
		divergenceAsOneHot=True,
		divergenceBins=20,
		divergenceBinSize=1.0,
		maxDivergencePoints=100,
		droneDrag=5.0,
		droneAngularDrag=7.0,
	)


def get_input_transforms(parameters: Dict[str, Any]):
	input_transform = []
	if np.isclose(float(parameters.get("enableNeuromorphicCamera", False)), 1.0):
		input_transform.append(
			Compose([
				Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
				threshold_image,
				Lambda(lambda t: torch.permute(t, (2, 0, 1))),
				Lambda(lambda t: torch.flatten(t, start_dim=1))
			])
		)
	input_transform.append(
		Compose([
			Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
		])
	)
	return input_transform


def create_curriculum(channel, n_lessons: int = 50, teacher=None):
	# if teachers is None:
	# 	teachers = [None for _ in range(n_lessons)]
	# elif not isinstance(teachers, list):
	# 	teachers = [teachers] * n_lessons
	# elif len(teachers) == 1:
	# 	teachers = [teachers[0] for _ in range(n_lessons)]
	# elif len(teachers) != n_lessons:
	# 	raise ValueError("Number of teachers must match number of lessons.")
	high_resolution = int(3*n_lessons/4)
	low_resolution = int(n_lessons - high_resolution)
	a, b, c = 1.0, 5.0, 10.0
	y_space = np.concatenate([np.linspace(a, b, high_resolution, endpoint=False), np.linspace(b, c, low_resolution)])

	lessons = [
		Lesson(
			f"Alt{str(y).replace('.', '_')}",
			channel,
			params=dict(droneMaxStartY=y),
			teacher=teacher if y <= 2.5 else None,
			completion_criteria=CompletionCriteria(measure='Rewards', min_lesson_length=5, threshold=0.9),
			teacher_strength=0.5**(less_idx+1) if y <= 2.5 else None,
		)
		for less_idx, y in enumerate(y_space)
	]
	return Curriculum(lessons=lessons)


def setup_environment(integration_time):
	build_path = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe"

	params_channel = EnvironmentParametersChannel()
	engine_config_channel = EngineConfigurationChannel()
	env = UnityEnvironment(
		file_name=build_path,
		seed=42,
		side_channels=[params_channel, engine_config_channel],
		no_graphics=False
	)
	engine_config_channel.set_configuration(EngineConfig(
		width=512, height=256, quality_level=1, time_scale=20.0, target_frame_rate=-1, capture_frame_rate=60
	))
	env_params = get_env_parameters(integration_time)
	sent_params = send_parameter_to_channel(params_channel, env_params)
	time.sleep(0.5)
	env.reset()
	time.sleep(0.5)
	return env, dict(params_channel=params_channel, engine_config_channel=engine_config_channel), env_params


def train_agent(env, integration_time, channels, env_params):
	mlp = MLPAgent(
		spec=env.behavior_specs[list(env.behavior_specs)[0]],
		behavior_name=list(env.behavior_specs)[0].split("?")[0],
		input_transform=get_input_transforms(env_params),
		n_hidden_neurons=[128, 128],
	)
	print(f"Training agent {mlp.name} on the behavior {mlp.behavior_name}.")
	print("\t behavior_spec: ", mlp.spec)
	print("\t input_sizes: ", mlp.input_sizes)
	academy = RLAcademy(
		env=env,
		agent=mlp,
		curriculum=create_curriculum(
			channels["params_channel"],
		),
		checkpoint_folder="checkpoints-mlp128x128-Input_divH-pbuffer",
	)
	hist = academy.train(
		n_iterations=int(1e3),
		# load_checkpoint_mode=LoadCheckpointMode.LAST_ITR,
		force_overwrite=True,
		save_freq=100,
		use_priority_buffer=True,
		max_seconds=1*60*60,
	)
	# hist.plot(show=True, figsize=(10, 6))


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	integration_time = 1
	env, channels, env_params = setup_environment(integration_time)
	train_agent(env, integration_time, channels, env_params)
	try:
		env.close()
	except:
		pass
