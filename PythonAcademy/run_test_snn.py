import os
import pickle
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
from PythonAcademy.src.snn_agent import SNNAgent
from PythonAcademy.src.spiking_layers import ALIFLayer, LIFLayer
from PythonAcademy.src.utils import send_parameter_to_channel, threshold_image, to_tensor


def get_env_parameters(int_time: int):
	return dict(
		n_agents=16,
		camFollowTargetAgent=False,
		droneMinStartY=1.0,
		droneMaxStartY=2.5,
		observationStacks=int_time,
		observationWidth=8,
		observationHeight=8,
		enableNeuromorphicCamera=True,
		enableCamera=False,
		usePositionAsInput=False,
		enableTorque=True,
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
	if np.isclose(float(parameters.get("useDivergenceAsInput", False)), 1.0):
		input_transform.append(
			Compose([
				Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
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
		width=726, height=512, quality_level=1, time_scale=20.0, target_frame_rate=-1, capture_frame_rate=60
	))
	env_params = get_env_parameters(integration_time)
	sent_params = send_parameter_to_channel(params_channel, env_params)
	time.sleep(0.5)
	env.reset()
	time.sleep(0.5)
	return env, dict(params_channel=params_channel, engine_config_channel=engine_config_channel), env_params


def test_agent(env, integration_time, channels, env_params):
	snn = SNNAgent(
		spec=env.behavior_specs[list(env.behavior_specs)[0]],
		behavior_name=list(env.behavior_specs)[0].split("?")[0],
		# n_hidden_neurons=[10, 5],
		n_hidden_neurons=[64, 8],
		# n_hidden_neurons=[128, 128],
		# n_hidden_neurons=[32, 16],
		# n_hidden_neurons=[16, 8],
		int_time_steps=integration_time,
		input_transform=get_input_transforms(env_params),
		use_recurrent_connection=False,
		hidden_layer_type=ALIFLayer,
		# hidden_layer_type=LIFLayer,
		# name="snn",
		# learn_beta=True,
	)
	print(f"Training device: {snn.device}")
	print(f"Training agent {snn.name} on the behavior {snn.behavior_name}.")
	print("\t behavior_spec: ", snn.spec)
	print("\t input_sizes: ", snn.input_sizes)
	# h = FgHeuristic(env, force_generate_trajectories=True, verbose=True, force_ratio=0.99, noise=0.1)
	# h.load_buffer()
	academy = RLAcademy(
		env=env,
		agent=snn,
		# checkpoint_folder="tr_data/checkpoints-alif10x5-Input_divH-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-lif10x5-Input_divH",
		# checkpoint_folder="tr_data/checkpoints-lif16x8-Input_eventCam8x8_divH-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-lif64x8-Input_eventCam8x8",
		# checkpoint_folder="tr_data/checkpoints-lif64x8-Input_eventCam8x8_divH",
		# checkpoint_folder="tr_data/checkpoints-alif64x8-Input_eventCam8x8_divH",
		# checkpoint_folder="tr_data/checkpoints-alif64x8-Input_eventCam8x8-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-alif16x8-Input_eventCam8x8_divH-pbuffer-learn_beta",
		# checkpoint_folder="tr_data/checkpoints-alif16x8-Input_eventCam8x8_divH",
		# checkpoint_folder="tr_data/checkpoints-alif16x8-Input_divH-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-lif64x8-Input_eventCam8x8-Out4",
		# checkpoint_folder="tr_data/checkpoints-lif64x8-Input_eventCam8x8_divH-Out4",
		# checkpoint_folder="tr_data/checkpoints-alif64x8-Input_eventCam8x8-Out4",
		checkpoint_folder="tr_data/checkpoints-lif64x8-Input_eventCam8x8_divH-Out4",
	)
	print(f"{academy.checkpoint_manager.checkpoints_meta_path = }")
	academy.check_and_load_state_from_academy_checkpoint(load_checkpoint_mode=LoadCheckpointMode.LAST_ITR)
	academy.training_histories.plot(
		save_path=f"figures/{os.path.basename(academy.checkpoint_folder)}.png",
		# lesson_idx=0,
		show=True,
	)
	_, cumulative_rewards = academy.generate_trajectories(n_trajectories=int(1e2))
	print(f"Cumulative rewards: {np.mean(cumulative_rewards):.3f} +/- {np.std(cumulative_rewards):.3f}")
	academy.close()


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	integration_time = 10
	env, channels, env_params = setup_environment(integration_time)
	test_agent(env, integration_time, channels, env_params)
	try:
		env.close()
	except:
		pass
