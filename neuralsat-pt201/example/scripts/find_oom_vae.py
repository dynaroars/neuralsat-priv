import torch.nn as nn
import torch
import time
import yaml
import os

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule
from util.misc.torch_cuda_memory import gc_cuda

from train.models.vae_naive import get_model

from .find_oom_vit import non_optim_bounds
    
if __name__ == "__main__":
        
    torch.manual_seed(0)
    device = 'cuda'
    shape = (1, 1, 16, 16)
    batch = 1
    method = 'backward'
    method = 'crown-optimized'
    extras = {}
    
    c = torch.Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, -1]]]).float()
    c = None
    
    config_path = 'train/config/mnist.yaml'
    config = yaml.safe_load(open(config_path))
    print(config)
    
    model = get_model(shape, latent_dim=64).to(device)
    
    # model = nn.Sequential(model, nn.Flatten(1))
    print(model)
    eps = 1e-4
    il = torch.randn(batch, *shape[1:]).to(device)
    iu = il + eps
    
    y = model(il)
    print(f'{il.shape=} {y.shape=}')
    
    if True:
        # end to end
        ol, ou = non_optim_bounds(model, shape, il, iu, c=c, device=device, method=method, conv_mode='patches', extras=extras)
        
    else:
        el, eu = non_optim_bounds(model.layers[0],    shape, il, iu, c=c, device=device, method=method, conv_mode='patches', extras=extras)
        gc_cuda()
        ol, ou = non_optim_bounds(model.layers[1], el.shape, el, eu, c=c, device=device, method=method, conv_mode='patches', extras=extras)