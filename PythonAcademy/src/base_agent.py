import json
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mlagents_envs.base_env import BehaviorSpec
from torch import nn
from torchvision.transforms import Compose, Lambda

from PythonAcademy.src.checkpoints_manager import CheckpointManager, LoadCheckpointMode
from PythonAcademy.src.utils import to_tensor
from PythonAcademy.src.wrappers import TensorActionTuple


class BaseAgent(torch.nn.Module):
	def __init__(
			self,
			spec: BehaviorSpec,
			behavior_name: str,
			name: str = "BaseAgent",
			checkpoint_folder: str = "checkpoints",
			device: torch.device = None,
			input_transform: Union[Dict[str, Callable], List[Callable]] = None,
			**kwargs
	):
		super(BaseAgent, self).__init__()
		self.spec = spec
		self.behavior_name = behavior_name
		self.name = name
		self.checkpoint_folder = checkpoint_folder
		self.kwargs = kwargs
		self.device = device
		if self.device is None:
			self._set_default_device_()

		if input_transform is None:
			input_transform = self.get_default_transform()
		if isinstance(input_transform, list):
			input_transform = {in_name: t for in_name, t in zip(self.input_sizes, input_transform)}
		self.input_transform: Dict[str, Callable] = input_transform
		self._add_to_device_transform_()

	@property
	def checkpoints_meta_path(self) -> str:
		full_filename = (
			f"{self.name}_{self.behavior_name}{CheckpointManager.SUFFIX_SEP}{CheckpointManager.CHECKPOINTS_META_SUFFIX}"
		)
		return f"{self.checkpoint_folder}/{full_filename}.json"

	def load_checkpoint(
			self,
			checkpoints_meta_path: Optional[str] = None,
			load_checkpoint_mode: LoadCheckpointMode = LoadCheckpointMode.BEST_ITR
	) -> dict:
		if checkpoints_meta_path is None:
			checkpoints_meta_path = self.checkpoints_meta_path
		with open(checkpoints_meta_path, "r+") as jsonFile:
			info: dict = json.load(jsonFile)
		path = CheckpointManager.get_save_name_from_checkpoints(info, load_checkpoint_mode)
		checkpoint = torch.load(path)
		self.load_state_dict(checkpoint[CheckpointManager.CHECKPOINT_STATE_DICT_KEY], strict=True)
		return checkpoint

	def get_default_transform(self) -> Dict[str, nn.Module]:
		return {
			in_name: Compose([
				Lambda(lambda a: to_tensor(a, dtype=torch.float32)),
			])
			for in_name in self.input_sizes
		}

	def apply_transform(self, obs: Sequence[Union[np.ndarray, torch.Tensor]]) -> Dict[str, torch.Tensor]:
		"""
		:param obs: shape: (observation_size, nb_agents, ...)
		:return: The input of the network with the same shape as the input.
		"""
		inputs = {
			obs_spec.name: torch.stack(
				[self.input_transform[obs_spec.name](obs_i) for obs_i in obs[obs_index]],
				dim=0
			)
			for obs_index, obs_spec in enumerate(self.spec.observation_specs)
		}
		return inputs

	def _add_to_device_transform_(self):
		for in_name, trans in self.input_transform.items():
			self.input_transform[in_name] = Compose([trans, Lambda(lambda t: t.to(self.device))])

	def _set_default_device_(self):
		self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	def forward(self, obs: Sequence[Union[np.ndarray, torch.Tensor]], **kwargs):
		raise NotImplementedError()

	def get_logits(self, obs: Sequence[Union[np.ndarray, torch.Tensor]]) -> torch.Tensor:
		raise NotImplementedError()

	def get_actions(
			self,
			obs: Sequence[Union[np.ndarray, torch.Tensor]],
			**kwargs
	) -> TensorActionTuple:
		"""
		Get the actions for the given observations.
		:param obs: The observations.
		False if the actions should be returned as torch tensors, used for training.
		:param kwargs: Other arguments.
		:return: The actions.
		"""
		batch_size = obs[0].shape[0]
		assert all([batch_size == o.shape[0] for o in obs]), "All observations must have the same batch size"
		return self.get_random_actions(batch_size)

	def get_random_actions(self, batch_size: int = 1) -> TensorActionTuple:
		return TensorActionTuple.from_numpy(self.spec.action_spec.random_action(batch_size))

	def soft_update(self, other: 'BaseAgent', tau: float = 1e-2) -> None:
		"""
		Copies the weights from the other network to this network with a factor of tau
		"""
		with torch.no_grad():
			for param, other_param in zip(self.parameters(), other.parameters()):
				param.data.copy_((1 - tau) * param.data + tau * other_param.data)

	def hard_update(self, other: 'BaseAgent') -> None:
		"""
		Copies the weights from the other network to this network
		"""
		with torch.no_grad():
			self.load_state_dict(other.state_dict())

	def to_onnx(self, in_viz=None):
		if in_viz is None:
			in_viz = torch.randn((1, self.input_size), device=self.device)
		torch.onnx.export(
			self,
			in_viz,
			f"{self.checkpoint_folder}/{self.name}.onnx",
			verbose=True,
			input_names=None,
			output_names=None,
			opset_version=11
		)

