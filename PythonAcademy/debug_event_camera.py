import pprint
import time
from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from mlagents_envs.base_env import BehaviorSpec
from tqdm.auto import tqdm
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from torchvision.transforms import Compose, Lambda

from PythonAcademy.src.curriculum import CompletionCriteria, Curriculum, Lesson
from PythonAcademy.src.heuristics import FgHeuristic, RandomHeuristic
from PythonAcademy.src.snn_agent import LoadCheckpointMode, SNNAgent
from PythonAcademy.src.utils import send_parameter_to_channel, threshold_image, to_tensor
from PIL import Image

import cv2


def div_to_bin_index(value: float, n_bins: int, bins_size: float) -> int:
	value /= bins_size
	n = int(n_bins/2)
	value = np.clip(value, -n, n)
	binIndex = int(value + n)
	return binIndex


def test_value_to_bin_index():
	values = np.arange(-10, 10, 1)
	bin_indexes = [div_to_bin_index(value, len(values), 1.0) for value in values]
	assert all([index == bin_indexes[index] for index, value in enumerate(values)])


if __name__ == '__main__':
	test_value_to_bin_index()
	build_path = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe"

	params_channel = EnvironmentParametersChannel()
	engine_config_channel = EngineConfigurationChannel()
	env = UnityEnvironment(
		file_name=build_path,
		seed=42,
		side_channels=[params_channel, engine_config_channel],
		no_graphics=False
	)
	engine_config = EngineConfig(1024, 728, 1, 1.0, -1, 60)
	engine_config_channel.set_configuration(engine_config)
	env_params = dict(
		n_agents=9,
		camFollowTargetAgent=True,
		droneMaxStartY=50.0,
		observationStacks=1,
		observationWidth=28,
		observationHeight=28,
		enableNeuromorphicCamera=True,
		enableCamera=False,
		divergenceAsOneHot=True,
		useDivergenceAsInput=True,
		usePositionAsInput=True,
		divergenceBins=20,
		divergenceBinSize=1.0,
		maxDivergencePoints=100,
	)
	sent_params = send_parameter_to_channel(params_channel, env_params)
	time.sleep(0.5)
	env.reset()

	behavior_name = list(env.behavior_specs)[0]
	spec: BehaviorSpec = env.behavior_specs[behavior_name]
	obs_specs_names = [o_spec.name for o_spec in spec.observation_specs]
	visual_obs_name = [o_name for o_name in obs_specs_names if 'camera' in o_name.lower()][0]
	print(f"obs_specs_names: {obs_specs_names}")
	print(f"Visual observation name: {visual_obs_name}")
	obs_traces = {k: [] for k in obs_specs_names}
	obs_shapes = {o_spec.name: o_spec.shape for o_spec in spec.observation_specs}
	pprint.pprint(obs_shapes, indent=4)
	n_steps = int(1e2)
	for i in tqdm(range(n_steps)):
		decision_steps, terminal_steps = env.get_steps(list(env.behavior_specs)[0])
		if len(decision_steps) > 0:
			for obs_name, obs in zip(obs_specs_names, decision_steps[0].obs):
				obs_traces[obs_name].append(obs)
			visual = np.asarray(obs_traces[visual_obs_name][-1]).squeeze()
			visual = threshold_image(visual)
			obs_traces[visual_obs_name][-1] = visual
			if env_params["observationStacks"] > 1:
				cv2.imshow('obs', cv2.resize(visual[..., 0], (256, 256)))
			else:
				cv2.imshow('obs', cv2.resize(visual, (256, 256)))
			env.set_actions(behavior_name, spec.action_spec.random_action(len(decision_steps)))
		env.step()
		key = cv2.waitKey(1)
		if key == 27:
			cv2.destroyAllWindows()
			break
	env.close()

	obs_traces = {k: np.asarray(v) for k, v in obs_traces.items()}
	for obs_name, obs_trace in obs_traces.items():
		min_value, max_value = np.min(obs_trace), np.max(obs_trace)
		print(f"{obs_name} - Min value: {min_value}, Max value: {max_value}")

	div_sensor_name = [o_name for o_name in obs_specs_names if 'divergence' in o_name.lower()][0]
	if env_params["divergenceAsOneHot"]:
		# make histogram ob observation
		plt.title(f"{div_sensor_name}")
		axis = 1 if env_params["observationStacks"] > 1 else -1
		print(f"{div_sensor_name}.args: {np.unique(np.argmax(obs_traces[div_sensor_name], axis=axis))}")
		plt.hist(np.argmax(obs_traces[div_sensor_name], axis=axis).squeeze(), bins=env_params["divergenceBins"])
		plt.show()
	else:
		indexes = [
				div_to_bin_index(div, env_params["divergenceBins"], env_params["divergenceBinSize"])
				for div in np.ravel(obs_traces[div_sensor_name])
			]
		print(f"{div_sensor_name}.args: {np.unique(indexes)}")
		plt.hist(indexes, bins=env_params["divergenceBins"])
		plt.show()

