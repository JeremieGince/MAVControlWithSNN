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


def create_curriculum(channel):
	lessons = [
		Lesson(
			f"Alt{str(y).replace('.', '_')}",
			channel,
			params=dict(droneMaxStartY=y),
			teacher=None,
		)
		for y in np.linspace(1.1, 10, num=10)
	]
	return Curriculum(lessons=lessons)


def setup_environment():
	build_path = "../MAVControlWithSNN/Builds/MAVControlWithSNN.exe"

	channel = EnvironmentParametersChannel()
	engine_config_channel = EngineConfigurationChannel()
	env = UnityEnvironment(
		file_name=build_path,
		seed=42,
		side_channels=[channel, engine_config_channel],
		no_graphics=True
	)
	engine_config_channel.set_configuration(EngineConfig.default_config())
	channel.set_float_parameter("batchSize", 4)
	channel.set_float_parameter("camFollowTargetAgent", False)
	channel.set_float_parameter("droneMaxStartY", 2.5)
	channel.set_float_parameter("observationStacks", 1)
	channel.set_float_parameter("observationWidth", 28)
	channel.set_float_parameter("observationHeight", 28)

	env.reset()
	return env


