import time
from typing import Any, Dict

import numpy as np
import torch
from tqdm.auto import tqdm
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from torchvision.transforms import Compose, Lambda

from PythonAcademy.src.curriculum import CompletionCriteria, Curriculum, Lesson
from PythonAcademy.src.heuristics import FgHeuristic, RandomHeuristic
from PythonAcademy.src.snn_agent import LoadCheckpointMode, SNNAgent
from PythonAcademy.src.utils import send_parameter_to_channel, threshold_image, to_tensor


def buff_trajectories(env: UnityEnvironment, n_trajectories: int = 1):
	env.reset()
	for _ in range(n_trajectories):
		while not env.get_steps(list(env.behavior_specs)[0])[1]:
			env.step()
	env.reset()


def get_env_parameters(int_time: int):
	return dict(
		batchSize=16,
		camFollowTargetAgent=False,
		droneMaxStartY=2.5,
		observationStacks=int_time,
		observationWidth=28,
		observationHeight=28,
		enableNeuromorphicCamera=True,
		enableCamera=False,
		divergenceAsOneHot=True,
		enableDivergence=False,
		usePositionAsInput=False,
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
			Lambda(lambda a: torch.from_numpy(a)),
		])
	)
	return input_transform


def create_curriculum(channel, teacher=None):
	lessons = [
		Lesson(
			f"Alt{str(y).replace('.', '_')}",
			channel,
			params=dict(droneMaxStartY=y),
			teacher=teacher if less_idx == 0 else None
		)
		for less_idx, y in enumerate(np.linspace(1.0, 10.0, num=100))
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
		width=512, height=512, quality_level=1, time_scale=20.0, target_frame_rate=-1, capture_frame_rate=60
	))
	env_params = get_env_parameters(integration_time)
	sent_params = send_parameter_to_channel(params_channel, env_params)
	time.sleep(0.5)
	env.reset()
	time.sleep(0.5)
	return env, dict(params_channel=params_channel, engine_config_channel=engine_config_channel), env_params


def train_agent(env, integration_time, channels, env_params):
	snn = SNNAgent(
		spec=env.behavior_specs[list(env.behavior_specs)[0]],
		behavior_name=list(env.behavior_specs)[0].split("?")[0],
		n_hidden_neurons=256,
		int_time_steps=integration_time,
		input_transform=get_input_transforms(env_params),
		use_recurrent_connection=False,
	)
	hist = snn.fit(
		env,
		n_iterations=int(1e4),
		curriculum=create_curriculum(
			channels["params_channel"],
			# teacher=FgHeuristic(env)
		),
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
	integration_time = 10
	env, channels, env_params = setup_environment(integration_time)
	train_agent(env, integration_time, channels, env_params)
	# for h in [FgHeuristic(env), RandomHeuristic(env)]:
	# 	print(f"{h.name} mean cumulative rewards: {np.mean([ex.reward for ex in h.buffer if ex.terminal]):.3f}")
	# values = np.unique(
	# 	np.asarray([
	# 		np.asarray(ex.obs)
	# 		for ex in tqdm(FgHeuristic(env, force_generate_trajectories=True, n_trajectories=1024).buffer)
	# 	])
	# )
	# min_value, max_value = np.min(values), np.max(values)
	# print(f"Min value: {min_value}, Max value: {max_value}")
	# lesson = Lesson(name="LowAltitude1DF", completion_criteria=0.9, teacher=h)
	try:
		env.close()
	except:
		pass




