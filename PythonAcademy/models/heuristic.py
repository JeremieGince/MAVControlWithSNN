import torch

from PythonAcademy.models.short_memory_model import SMModel


class Heuristic(SMModel):
	def __init__(self,
	             in_shape: Union[Tuple, List, Iterable],
                 out_shape: Union[Tuple, List, Iterable],
                 n_hidden_layers: int = 3,
                 hidden_dim: int = 64,
                 memory_size: int = 10,
                 **kwargs):
		super(Heuristic, self).__init__()
		self.a = torch.nn.Parameter(torch.randn(()))
		self.b = torch.nn.Parameter(torch.randn(()))

	def forward(self, x):
		return self.a * x + self.b









