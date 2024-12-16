import torch.nn as nn
import torch
import time
import yaml
import os
import sys
import random

from util.misc.torch_cuda_memory import gc_cuda
from train.models.resnet.resnet import *
from decomposer.utils import PytorchWrapper

from .find_oom_vit import non_optim_bounds
    
from setting import Settings


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

if __name__ == "__main__":
        
    Settings.setup(None)
        
    torch.manual_seed(sys.argv[-1])
    random.seed(sys.argv[-1])
    
    device = 'cuda'
    shape = (1, 3, 32, 32)
    batch = 1
    method = 'backward'
    # method = 'crown-optimized'
    # extras = {'sparse_intermediate_bounds': False}
    extras = {}
    
    # c = torch.Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, -1]]]).float()
    # print(c.shape)
    indices = torch.arange(0, 1)
    c = torch.nn.functional.one_hot(indices, num_classes=10)[None].float().to(device)
    print(c.shape)
    # c = None
    
    
    model_name = 'resnet3'
    n_layers = 2
    eps = 0.2
    
    model = eval(model_name)().to(device)
    
    
    
    # model = torch.load('example/generated_benchmark/cifar10_2/eps_in_0.0020_eps_out_0.0050/net/cifar10_2.pth').to(device)
    model.eval()
    # model = nn.Sequential(model, nn.Flatten(1))
    print(model)
    print(get_model_params(model))
    # eps = 2.0
    il = torch.randn(batch, *shape[1:]).to(device)
    iu = il + eps
    
    y = model(il)
    
    prefix = PytorchWrapper(model.layers[:n_layers])
    suffix = PytorchWrapper(model.layers[n_layers:])
    
    print(f'{il.shape=} {y.shape=}')
    
    # print(prefix)
    # print(suffix)
    y1 = suffix(prefix(il))
    assert torch.norm(y1 - y) < 1e-6

    if int(sys.argv[1]):
        # end to end
        ol, ou = non_optim_bounds(model, shape, il, iu, c=c, device=device, method=method, conv_mode='patches', extras=extras)
        
    else:
        el, eu = non_optim_bounds(nn.Sequential(model.layers[0], nn.Flatten(1)),    shape, il, iu, c=None, device=device, method=method, conv_mode='patches', extras=extras)
        out_shape = model.layers[0](il).shape
        gc_cuda()
        ol, ou = non_optim_bounds(nn.Sequential(model.layers[1], nn.Flatten(1)), out_shape, el.view(out_shape), eu.view(out_shape), c=c, device=device, method=method, conv_mode='patches', extras=extras)
    
    print()
    print(f'{il.sum().item()=} {iu.sum().item()=}')
    print(f'{ol=}')