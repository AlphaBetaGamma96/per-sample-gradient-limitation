import torch
import torch.nn as nn

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

hooks = {}

#forward-pre hook and backward full hook
def _save_input(module, input):
 hooks['a'] = input[0]

def _save_output(module, grad_input, grad_output):
 hooks['e'] = grad_output[0]

batch = 1
num_input = 1
num_output = 6

X = torch.randn(batch, num_input)                               #shape 100, 4 (100 samples, 4 inputs per sample)

fc = nn.Linear(num_input, num_output, bias=False)                      #weight shape is (6,4), and no bias for easier example
act_func=nn.Tanh()

#very simple network
def network(X):
  Y=fc(X)                                               #output shape of layer is (100, 6)
  Y=act_func(Y)                                         #non-linear activation function
  output = Y.pow(2).sum(dim=-1)                #shape of 100, (individual loss of each sample)
  return output

#loss function to get per-sample gradients of
def loss_function(X):
  #loss_per_sample = network(X) #this loss function works 
  #loss_per_sample = network(X) + laplacian(network, X) #this loss function fails
  loss_per_sample = network(X) + laplacian_from_hessian(network, X) #this works (but only for one sample)
  return loss_per_sample

#register forward/backward hooks
fc.register_full_backward_hook(_save_output)
fc.register_forward_pre_hook(_save_input)

#==============================================================================#
"""
Compare batch method (which uses hooks) versus doing a sequential method which 
sequentially iterates over all samples and compute a single .backward() call for 
each sample to get a batch of per-sample gradients.
Both methods use the same loss function, although if the loss function contains 
this laplacian operator it fails and gives different values for some reason.
"""

#batch method
loss_per_sample = loss_function(X)
total_loss = torch.mean(loss_per_sample, dim=0)       #loss is scalar
total_loss.backward()

a, e = hooks['a'], hooks['e']

all_grad_weight = torch.einsum("bi,bj->bij", e, a) * a.shape[0]


#sequential method
grad_weight_list = []
fc.weight.grad.zero_()

for xi in X:
  loss_per_sample = loss_function(xi.unsqueeze(0))

  loss_per_sample.backward()
  grad_weight_list.append(fc.weight.grad.clone())
  fc.weight.grad.zero_() #clear grad cache
  
grad_weight_list = torch.stack(grad_weight_list)

#check both methods!
print(all_grad_weight, grad_weight_list)
print("allclose check: ",torch.allclose(all_grad_weight, grad_weight_list)) 
#returns True for first loss-function, False for second loss-function.
