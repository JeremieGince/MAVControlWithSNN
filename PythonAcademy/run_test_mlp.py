import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
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
		n_agents=16,
		camFollowTargetAgent=False,
		droneMaxStartY=2.5,
		observationStacks=int_time,
		observationWidth=8,
		observationHeight=8,
		enableNeuromorphicCamera=True,
		enableCamera=False,
		enableTorque=True,
		enableDisplacement=False,
		usePositionAsInput=False,
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
	return env, dict(params_channel=params_channel, engine_config_channel=engine_config_channel), env_params


def test_agent(env, integration_time, channels, env_params):
	mlp = MLPAgent(
		spec=env.behavior_specs[list(env.behavior_specs)[0]],
		behavior_name=list(env.behavior_specs)[0].split("?")[0],
		n_hidden_neurons=[128, 128],
		# n_hidden_neurons=[10, 5],
	)
	print(f"Training device: {mlp.device}")
	print(f"Training agent {mlp.name} on the behavior {mlp.behavior_name}.")
	print("\t behavior_spec: ", mlp.spec)
	print("\t input_sizes: ", mlp.input_sizes)
	academy = RLAcademy(
		env=env,
		agent=mlp,
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_divH-pbuffer",
		# checkpoint_folder="tr_data/checkpoints_mlp_Input_pos-prioritybuffer",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_pos",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_posV",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_divH-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-mlp10x5-Input_divH-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-mlp10x5-Input_div",
		# checkpoint_folder="tr_data/checkpoints_mlp128x128-Input_posV_angV_div-env_torque-pbuffer",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_eventCam8x8",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_eventCam8x8_divH",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_posV_angV-Out4",
		# checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_eventCam8x8-Out4",
		checkpoint_folder="tr_data/checkpoints-mlp128x128-Input_eventCam8x8_divH-Out4",
	)
	print(f"{academy.checkpoint_manager.checkpoints_meta_path = }")
	academy.check_and_load_state_from_academy_checkpoint(load_checkpoint_mode=LoadCheckpointMode.BEST_ITR)
	academy.training_histories.plot(
		save_path=f"figures/{os.path.basename(academy.checkpoint_folder)}.png",
		# lesson_idx=0,
		show=True,
	)
	_, cumulative_rewards = academy.generate_trajectories(n_trajectories=int(1e2))
	print(f"Cumulative rewards: {np.mean(cumulative_rewards):.3f} +/- {np.std(cumulative_rewards):.3f}")
	academy.close()
	# plt.hist(cumulative_rewards, bins=10)
	# plt.show()


if __name__ == '__main__':
	# mlagents-learn config/Landing_wo_demo.yaml --run-id=eventCamLanding --resume
	integration_time = 1
	env, channels, env_params = setup_environment(integration_time)
	test_agent(env, integration_time, channels, env_params)
	try:
		env.close()
	except:
		pass
