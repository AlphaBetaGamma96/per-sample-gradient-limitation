import torch
import torch.nn as nn

from torch import Tensor
from typing import List, Optional

torch.manual_seed(0)

def laplacian(net, xs):
  """
  Computes the laplacian (i.e. Trace of Hessian) of a network with respect to its inputs
  """
  xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
  xs_flat = torch.stack(xis, dim=1)

  ys = net(xs_flat.view_as(xs))

  ones = torch.ones_like(ys)

  (dy_dxs, ) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)

  
  lay_ys = sum(torch.autograd.grad(dy_dxi, xi, ones, retain_graph=True, create_graph=True)[0] \
                for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis))))               
  )

  return lay_ys 

def laplacian_from_hessian(net, xs):
  hess = torch.autograd.functional.hessian(net, xs).reshape(xs.shape[-1], xs.shape[-1])
  return hess.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)

#decorator for counting calls to given function
def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped

hooks = {}

@counted
#forward-pre hook and backward full hook
def _save_input(module, input):
 hooks['a'] = input[0]

@counted
def _save_output(module, grad_input, grad_output):
 #print("e: ",module, grad_output[0], grad_input)
 hooks['e'] = grad_output[0]

batch = 100
num_input = 1
num_output = 6

X = torch.randn(batch, num_input)                               #shape 100, 4 (100 samples, 4 inputs per sample)

fc = nn.Linear(num_input, num_output, bias=False)                      #weight shape is (6,4), and no bias for easier example
act_func=nn.Tanh()

fc.register_full_backward_hook(_save_output)
fc.register_forward_pre_hook(_save_input)

#very simple network
def network(X):
  Y=fc(X)                                               #output shape of layer is (100, 6)
  Y=act_func(Y)                                         #non-linear activation function
  output = Y.pow(2).sum(dim=-1)                #shape of 100, (individual loss of each sample)
  return output

@torch.jit.script
def sumit(inp: List[Optional[torch.Tensor]]):
  elt = inp[0]
  if elt is None:
      raise RuntimeError("blah")
  base = elt
  for i in range(1, len(inp)):
    next_elt = inp[i]
    if next_elt is None:
        raise RuntimeError("blah")
    base = base + next_elt
  return base

@torch.jit.script
def laplacian_jit(xs):
  xis = [xi.requires_grad_() for xi in xs.flatten(start_dim=1).t()]
  xs_flat = torch.stack(xis, dim=1)
  ys = network(xs_flat.view_as(xs))

  ones = torch.ones_like(ys)
  grad_outputs = torch.jit.annotate(List[Optional[Tensor]], [])
  grad_outputs.append(ones)
  result = torch.autograd.grad([ys], [xs_flat], grad_outputs, retain_graph=True, create_graph=True)
  dy_dxs = result[0]
  if dy_dxs is None:
      raise RuntimeError("blah")

  generator_as_list = [dy_dxs[..., i] for i in range(len(xis))]
  lap_ys_components = [torch.autograd.grad([dy_dxi], [xi], grad_outputs, retain_graph=True, create_graph=False)[0] \
                          for xi, dy_dxi in zip(xis,generator_as_list)]

  lap_ys = sumit(lap_ys_components)

  return lap_ys
  
#==============================================================================#

#loss function to get per-sample gradients of
def loss_function(X):
  #loss_per_sample = network(X) + laplacian(network, X) #this loss function fails
  loss_per_sample = network(X) + laplacian_jit(X) #this works with batch
  return loss_per_sample
  
#batch method
loss_per_sample = loss_function(X)
total_loss = torch.mean(loss_per_sample, dim=0)       #loss is scalar
total_loss.backward()

a, e = hooks['a'], hooks['e']

batch_all_grad_weight = torch.einsum("bi,bj->bij", e, a) * a.shape[0]

print("per-sample (batch method)")
print("_save_input: ",_save_input.calls)
print("_save_output: ",_save_output.calls)

#reset counters
_save_input.calls = 0
_save_output.calls = 0

#sequential method
grad_weight_list = []
fc.weight.grad.zero_()

for xi in X:
  loss_per_sample = loss_function(xi.unsqueeze(0))

  loss_per_sample.backward()
  grad_weight_list.append(fc.weight.grad.clone())
  fc.weight.grad.zero_() #clear grad cache
  
sequential_all_grad_weight = torch.stack(grad_weight_list)

print("per-sample (sequential method)")
print("_save_input: ",_save_input.calls)
print("_save_output: ",_save_output.calls)

print("allclose check: ",torch.allclose(batch_all_grad_weight, sequential_all_grad_weight)) 
