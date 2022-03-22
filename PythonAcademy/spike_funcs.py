from typing import Any

import torch


class HeavisideSigmoidApprox(torch.autograd.Function):
	scale = 100.0  # controls steepness of surrogate gradient
	threshold = 0.0

	@staticmethod
	def forward(ctx: Any, inputs):
		"""
		In the forward pass we compute a step function of the input Tensor
		and return it. ctx is a context object that we use to stash information which
		we need to later backpropagate our error signals. To achieve this we use the
		ctx.save_for_backward method.
		"""
		ctx.save_for_backward(inputs)
		out = torch.zeros_like(inputs)
		out[inputs >= HeavisideSigmoidApprox.threshold] = 1.0
		return out

	@staticmethod
	def backward(ctx: Any, grad_outputs):
		"""
		In the backward pass we receive a Tensor we need to compute the
		surrogate gradient of the loss with respect to the input.
		Here we use the normalized negative part of a fast sigmoid
		as this was done in Zenke & Ganguli (2018).
		"""
		input, = ctx.saved_tensors
		grad_input = grad_outputs.clone()
		grad = grad_input / (HeavisideSigmoidApprox.scale * torch.abs(input) + 1.0) ** 2
		return grad



