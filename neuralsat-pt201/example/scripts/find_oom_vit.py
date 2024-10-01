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
from train.models.vit.vit import *
from decomposer.dec_verifier import PytorchWrapper
from abstractor.params import get_initialize_opt_params

EXTRAS_VIT = {
    'sparse_intermediate_bounds': False,
    # 'sparse_conv_intermediate_bounds': True,
    # 'matmul': {'share_alphas': True},
    # 'disable_optimization': ['Exp']
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
    
    l, u, aux_reference_bounds = abstract.init_alpha(
        x=(new_x,), 
        share_alphas=True, 
        c=c, 
        bound_upper=True,
    )
    if verbose:
        print(f'{l.shape=}')
        print(f'{l=}')
        print(f'{u=}')
    
    if method == 'backward':
        return l, u
    
    abstract.set_bound_opts(get_initialize_opt_params(lambda x: False))
    abstract.set_bound_opts({'optimize_bound_args': {'iteration': 10}})
    
    l, u = abstract.compute_bounds(
        x=(new_x,), 
        C=c,
        bound_upper=True, 
        method='crown-optimized',
        aux_reference_bounds=aux_reference_bounds, 
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
    # d = 'cpu'
    shape = (1, 3, 32, 32)
    # shape = (1, 1, 28, 28)
    batch = 1
    
    c = torch.Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]).float()
    # c = None

    model_name = 'vit_toy'
    vit_orig = eval(model_name)(positional_embedding='none').to(d)
    vit_orig.eval()
    # vit.load_state_dict(torch.load(f'train/weights/{model_name}.pt'))
    
    if 0:
        shape = vit_orig.tokenizer(torch.randn(shape, device=d)).size()
        print(f'new {shape=}')
        # exit()
        vit = PytorchWrapper(vit_orig.classifier).to(d)
    else:
        vit = vit_orig.to(d)
    
    
    torch.onnx.export(
        vit_orig.cpu(),
        torch.randn(shape),
        'train/weights/toy.onnx',
        export_params=True,
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},    # Variable batch size
            'output': {0: 'batch_size'}    # Variable batch size
        }
    )
    
    # vit = nn.Sequential(
    #     nn.Flatten(1),
    #     nn.Linear(3072, 24),
    #     nn.ReLU(),
    #     nn.Linear(24, 32),
    #     nn.ReLU(),
    #     nn.Linear(32, 10),
    # )
    
    print(vit)
    # exit()
    get_model_params(vit)
    method = 'backward'
    # method = 'forward'
    method = 'crown-optimized'
    # method = 'ibp'
    
    # il = torch.randn(batch, *shape[1:]).to(d)
    # iu = il + eps
    
    eps = 0.01
    
    il = torch.randn(batch, *shape[1:]).to(d)
    iu = il + eps
    
    vit.train()
    vit = vit.to(d)
    print(f'{il.sum()=} {iu.sum()=}')
    print(vit(il))
    # il = vit_orig(il)
    ol, ou = non_optim_bounds(vit, shape, il, iu, c=c, device=d, method=method, conv_mode='matrix')
    exit()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./train/data/',
        train=False,
        transform=transform,
        download=False
    )
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    # print(dataset[0])
    # exit()
    for idx, (x, _) in enumerate(tqdm.tqdm(dataloader)):
        # il = dataset[int(sys.argv[1])][0][None].to(d)
        # if idx != 174:
        #     continue
        # il = x.to(d) - eps
        # iu = x.to(d) + eps
        
        # il = vit_orig.tokenizer(il)
        # iu = vit_orig.tokenizer(iu)
            
        il = torch.randn(batch, *shape[1:]).to(d)
        iu = il + eps
        
        # print(f'{il.shape=}')
        # print(vit(il))
        # il = vit_orig(il)
        ol, ou = non_optim_bounds(vit, shape, il, iu, c=c, device=d, method=method, conv_mode='matrix')
        exit()
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