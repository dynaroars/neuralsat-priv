import onnxruntime as ort
import onnx2pytorch
import numpy as np
import logging
import torch
import tqdm
import onnx
import io

from util.misc.logger import logger


from verifier.objective import Objective, DnfObjectives
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx, decompose_onnx, decompose_pytorch


from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedModule, BoundedTensor


def get_hidden_bounds(self, device):
    lower_bounds, upper_bounds = {}, {}
    # print(list(set(self.layers_requiring_bounds + self.split_nodes)))
    for layer in list(set(self.layers_requiring_bounds + self.split_nodes)):
        lower_bounds[layer.name] = layer.lower.detach().to(device)
        upper_bounds[layer.name] = layer.upper.detach().to(device)

    return lower_bounds, upper_bounds


def print_w_b(self):
    for layer in self.children():
        if hasattr(layer, 'weight'):
            print(layer)
            print('\t[+] w:', layer.weight.data.flatten())
            print('\t[+] b:', layer.bias.data.flatten())
            print()
            
def extract_instance(net_path, vnnlib_path):
    vnnlibs = read_vnnlib(vnnlib_path)
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives


def get_activation_shape(name, result):
    def hook(model, input, output):
        result[name] = output.shape
    return hook


def split_model(model, split_layer: int, input_shape=None, device='cpu'):
    prefix_layers = []
    suffix_layers = []
    layer_idx = 0
    for layer in model.children():
        if layer_idx < split_layer:
            prefix_layers.append(layer)
        else:
            suffix_layers.append(layer)
            
        if isinstance(layer, torch.nn.ReLU):
            layer_idx += 1
    
    prefix_layers = torch.nn.Sequential(*prefix_layers).to(device)
    suffix_layers = torch.nn.Sequential(*suffix_layers).to(device)
    if input_shape:
        x = torch.ones(input_shape, device=device)
        assert torch.equal(suffix_layers(prefix_layers(x)), model(x))
    return prefix_layers, suffix_layers
        
@torch.no_grad()
def sampling(model, lower_bounds, upper_bounds, n_sample=100, device='cpu'):
    x = (upper_bounds - lower_bounds) * torch.rand(n_sample, *lower_bounds.shape[1:], device=device) + lower_bounds
    # print(x)
    output = []
    for layer in model.children():
        if isinstance(layer, torch.nn.ReLU):
            output.append(x.clone())
        x = layer(x)
    return output
        

def run(onnx_name, vnnlib_name, timeout, device):
    print('[+] Step 1: Load model')
    model, input_shape, objectives = extract_instance(onnx_name, vnnlib_name)
    model.to(device)
    # print(model)
    # print_w_b(model)
    # print(f'{input_shape=}')
    
    if 1:
        print('\n\n[+] Step 2: Load property')
        lower_bounds = objectives.lower_bounds.view(-1, *input_shape[1:]).to(device)
        upper_bounds = objectives.upper_bounds.view(-1, *input_shape[1:]).to(device)
        cs = objectives.cs.to(device)
        print(f'{lower_bounds=}')
        print(f'{upper_bounds=}')
        
        print('\n\n[+] Step 3: Init abstractor for the original model')
        # property
        input_prop = BoundedTensor(lower_bounds, PerturbationLpNorm(x_L=lower_bounds, x_U=upper_bounds)).to(device)
        
        # abstractor
        net = BoundedModule(
            model=model, 
            global_input=torch.zeros(input_shape, device=device),
            bound_opts={'conv_mode': 'matrix', 'verbosity': 0,},
            device=device,
            verbose=False,
        )
        # print(net)
        net.eval()
        print(f'{net(input_prop)=}')
        net.get_split_nodes()
        
        # with torch.no_grad():
        if 1:
            lb, ub = net.compute_bounds(x=(input_prop,), method='backward', bound_upper=True)
            print(f'{lb=}')
            print(f'{ub=}')
        print()
        lbs, ubs = get_hidden_bounds(net, device)
        for i, k in enumerate(net.split_nodes):
            print('\t- Layer', i, lbs[k.name])
            print('\t         ', ubs[k.name])
            
    # Samples
    print('\n\n[+] Step 4: Sampling')
    sample_outputs = sampling(
        model=model, 
        lower_bounds=lower_bounds, 
        upper_bounds=upper_bounds, 
        n_sample=1000, 
        device=device,
    )
    
    ref_bounds = {
        k: [lbs[k], ubs[k]] for k in lbs
    }
    for i, o in enumerate(sample_outputs):
        ol = o.min(0).values
        ou = o.max(0).values
        print('lower:', o.shape, ol)
        print('upper:', o.shape, ou)
        
        for ni, (nl, nu) in enumerate(zip(ol, ou)):
            print(i, ni, nl, nu)
            if nl >= 0:
                ref_bounds[net.split_nodes[i].name][0][0][ni] = nl
            elif nu <= 0:
                ref_bounds[net.split_nodes[i].name][1][0][ni] = nu
        
        ref_bounds[net.split_nodes[i].name] = [ol, ou]
        print()

    # with torch.no_grad():
    
    
    for i, k in enumerate(net.split_nodes):
        print('\t- Layer', i, ref_bounds[k.name][0])
        print('\t         ', ref_bounds[k.name][1])
        
    if 1:
        lb, ub = net.compute_bounds(
            x=(input_prop,), 
            C=cs,
            method='crown-optimized', 
            bound_upper=False, 
            reference_bounds=ref_bounds,
        )
        print(f'{lb=}')
    
    
def execute(net, shape, lower, upper, device, method='backward', cs=None, verbose=True):
    prop = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)
    
    abstract = BoundedModule(
        model=net, 
        global_input=torch.zeros(shape, device=device),
        bound_opts={'conv_mode': 'matrix', 'verbosity': 0,},
        device=device,
        verbose=False,
    )
    abstract.eval()
    abstract(prop)
    abstract.get_split_nodes()

    lb, ub = abstract.compute_bounds(x=(prop,), method=method, C=cs, bound_upper=cs is None)
    if verbose:
        print(f'[{method}] {lb = }')
        print(f'[{method}] {ub = }')
    return abstract, lb, ub


def split_model(net_path, vnnlib_path, split_idx):
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    if 0:
        prefix_onnx_byte, suffix_onnx_byte = decompose_onnx(net_path, split_idx)
    else:
        prefix_onnx_byte, suffix_onnx_byte = decompose_pytorch(model, input_shape, split_idx)
        
    print(model)
    prefix, input_shape_prefix, _, _ = parse_onnx(prefix_onnx_byte)
    print(prefix)
    suffix, input_shape_suffix, _, _ = parse_onnx(suffix_onnx_byte)
    print(suffix)
    return (model, input_shape), (prefix, input_shape_prefix), (suffix, input_shape_suffix), objectives
    

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG)
    net_name = 'example/onnx/mnistfc-medium-net-554.onnx'
    vnnlib_name = 'example/vnnlib/test.vnnlib'

    net_name = 'example/onnx/mnistfc-medium-net-151.onnx'
    vnnlib_name = 'example/vnnlib/prop_2_0.03.vnnlib'
    
    # net_name = 'example/onnx/motivation_example_3986.onnx'
    # vnnlib_name = 'example/vnnlib/motivation_example_3986.vnnlib'
    
    device = 'cpu'
    (model, input_shape), (prefix, input_shape_prefix), (suffix, input_shape_suffix), objectives = split_model(net_name, vnnlib_name, 3)
    
    x = torch.randn(17, *input_shape[1:])
    for i in tqdm.tqdm(range(100)):
        y1 = model(x)
        y2 = suffix(prefix(x))
        assert torch.equal(y1, y2)
    print('Matched')
    print()
        
    method = 'crown-optimized'        
    # method = 'backward'        
    input_lower = objectives.lower_bounds[0:1].view(-1, *input_shape[1:]).to(device)
    input_upper = objectives.upper_bounds[0:1].view(-1, *input_shape[1:]).to(device)
    
    cs = objectives.cs.transpose(0, 1).to(device)
    # cs = None
    
    print('abstract full')
    abstract_full, lb_full, ub_full = execute(model, input_shape, input_lower, input_upper, device, method=method, cs=cs)
    # print(abstract_full)
    print()
    
    # print('abstract prefix')
    abstract_pre, lb_pre, ub_pre = execute(prefix, input_shape_prefix, input_lower, input_upper, device, method=method, verbose=False)
    # print()
    
    print('abstract suffix')
    abstract_suf, lb_suf, ub_suf = execute(suffix, input_shape_suffix, lb_pre, ub_pre, device, method=method, cs=cs)
    # print(abstract_suf)
    print()
    