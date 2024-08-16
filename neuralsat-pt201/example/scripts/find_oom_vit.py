import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import traceback
import torch
import time
import tqdm
import sys

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule
from util.misc.torch_cuda_memory import gc_cuda
from train.models.vit import get_model
from train.models.vit_3 import vit_2_4, vit_7_4

EXTRAS_VIT = {
    'sparse_intermediate_bounds': False,
    # 'sparse_conv_intermediate_bounds': True,
}

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params = }')
    return total_params

def non_optim_bounds(net, input_shape, lower, upper, verbose=True, c=None, device='cpu', method='backward', conv_mode='patches', extras=EXTRAS_VIT):
    new_x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)

    print(f'Computing bounds using {method=} {c=} {conv_mode=} {extras=}')
    abstract = BoundedModule(
        model=net, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts={'conv_mode': conv_mode, 'verbosity': 0, **extras},
        # bound_opts={'conv_mode': 'matrix', 'verbosity': 0},
        device=device,
        verbose=False,
    )
    print(f'{abstract(new_x).shape=}')
    
    abstract.set_bound_opts({'optimize_bound_args': {'iteration': 2}})
    
    l, u = abstract.compute_bounds(
        x=(new_x,), 
        C=c,
        bound_upper=True, 
        method=method,
    )
    
    if verbose:
        print(f'{l.shape=}')
        print(f'{l=}')
        print(f'{u=}')
    
    assert torch.all(l <= u)
    del abstract
    return l, u
    
    
if __name__ == "__main__":
        
    seed = int(sys.argv[1])
    print(f'{seed=}')
    torch.manual_seed(seed)
    d = 'cuda'
    shape = (1, 3, 32, 32)
    shape = (1, 1, 28, 28)
    batch = 1
    
    c = torch.Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, -1]]]).float()
    c = None

    # vit = _parse_onnx(net_path)[0].to(d)
    vit = get_model(
        input_shape=shape,
        depth=4, 
        num_heads=4, 
        patch_size=14, 
        embed_dim=256, 
        weights=True,
    ).to(d)
    
    # vit = vit_2_4().to(d)
    
    vit.eval()
    print(vit)
    get_model_params(vit)
    # method = 'backward'
    method = 'crown-optimized'
    # method = 'ibp'
    
    # il = torch.randn(batch, *shape[1:]).to(d)
    # iu = il + eps
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./train/data/',
        train=False,
        transform=transform,
        download=False
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # print(dataset[0])
    eps = 0.005
    # exit()
    for idx, (x, _) in enumerate(tqdm.tqdm(dataloader)):
        # il = dataset[int(sys.argv[1])][0][None].to(d)
        # if idx != 174:
        #     continue
        il = x.to(d) - eps
        iu = x.to(d) + eps
        # print(f'{il.shape=}')
        # print(vit(il))
        # ol, ou = non_optim_bounds(vit, shape, il, iu, c=c, device=d, method=method, conv_mode='matrix')
        # exit()
        try:
            if False:
                # end to end
                ol, ou = non_optim_bounds(vit, shape, il, iu, c=c, device=d, method=method, conv_mode='matrix')
            else:
                unflattened_shape = vit.layers[0](il).shape
                prefix = torch.nn.Sequential(vit.layers[0], nn.Flatten(1))
                el, eu = non_optim_bounds(prefix, shape, il, iu, verbose=False, c=None, device=d, method=method, conv_mode='matrix')
                gc_cuda()
                el = el.view(unflattened_shape).contiguous()
                eu = eu.view(unflattened_shape).contiguous()
                # el = torch.randn(unflattened_shape).to(d)
                # eu = el + eps
                ol, ou = non_optim_bounds(vit.layers[1], unflattened_shape, el, eu, c=c, device=d, method=method, conv_mode='matrix')

        except AssertionError:
            traceback.print_exc()
            print(f'Failed with {idx=}')
            continue

        
        print(f'Passed with {idx=}')
        exit()