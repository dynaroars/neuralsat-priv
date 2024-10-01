
from pathlib import Path
import torch.nn as nn
import random
import torch
import time
import os

from verifier.verifier import Verifier 
from util.spec.read_vnnlib import read_vnnlib
from util.network.read_onnx import parse_onnx
from verifier.objective import Objective, DnfObjectives
from util.misc.logger import logger
from setting import Settings


def extract_instance(net_path, vnnlib_path):
    vnnlibs = read_vnnlib(vnnlib_path)
    pytorch_path = net_path[:-5] + '.pth'
    if os.path.exists(pytorch_path):
        print(f'Loading {pytorch_path=}')
        model = torch.load(pytorch_path)
        input_shape = (1, 3, 32, 32)
        is_nhwc = False
    else:
        model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
        
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
    objectives = DnfObjectives(objectives, input_shape=input_shape, is_nhwc=is_nhwc)

    return model, input_shape, objectives

    
def test_1():
    net_path = 'example/backup/motivation_example_159.onnx'
    vnnlib_path = 'example/backup/motivation_example_159.vnnlib'
    
    # net_path = 'example/onnx/mnist-net_256x2.onnx'
    # vnnlib_path = 'example/vnnlib/prop_1_0.05.vnnlib'
    
    device = 'cpu'
    logger.setLevel(1)
    # Settings.setup(args=None)
    Settings.setup_test()
    
    print(Settings)
    
    print('Running test with', net_path, vnnlib_path)
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    print(model)

    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    
    # stable, unstable, lbs, ubs = verifier.compute_stability(objectives)
    # print('stable:', stable)
    # print('unstable:', unstable)
    
    preconditions = [
        # {'/input': (torch.tensor([0]), torch.tensor([-1.]), torch.tensor([0.])), '/input.3': ([], [], [])},
        {'/input': (torch.tensor([0, 1]), torch.tensor([1., 1.]), torch.tensor([0., 0.])), '/input.3': (torch.tensor([0]), torch.tensor([1.]), torch.tensor([0.]))},
        {'/input': (torch.tensor([0, 1]), torch.tensor([ 1., -1.]), torch.tensor([0., 0.])), '/input.3': (torch.tensor([0, 1]), torch.tensor([-1.,  1.]), torch.tensor([0., 0.]))},
    ]
    # preconditions = []
    
    print(preconditions)
    
    verifier.verify(objectives, preconditions=preconditions)
    print('status:', verifier.status)
    print('unsat core:', verifier.get_unsat_core())
    
    # for c in verifier._get_learned_conflict_clauses():
    #     print(c)
    # print('lbs:', lbs)
    # print('ubs:', ubs)
    
    print(verifier.get_stats())


def test_2():
    net_path = 'example/mnist-net_256x2.onnx'
    vnnlib_path = Path('example/prop_1_0.03.vnnlib')
    device = 'cuda'

    print('\n\nRunning test with', net_path, vnnlib_path)
    
    # preconditions = [eval(l.replace('tensor', 'torch.tensor')) for l in open('log.txt').read().strip().split('\n')][:400]
    # print(preconditions)
    preconditions = []
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    Settings.setup(args=None)
    print(Settings)
    logger.setLevel(1)
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=200,
        device=device,
    )
    
    
    status = verifier.verify(objectives, preconditions=preconditions)
    print('status:', status, verifier.status)
    print('unsat core')
    from pprint import pprint
    print(verifier.get_unsat_core())
    
    # for c in verifier._get_learned_conflict_clauses():
    #     print(c)



def generate_simple_specs(dnf_pairs, n_outputs):
    all_cs = []
    all_rhs = []
    for cnf_pairs in dnf_pairs:
        cs = []
        rhs = []
        for output_i, rhs_i, direction in cnf_pairs:
            c = torch.zeros(n_outputs)
            r = torch.tensor(rhs_i)
            c[output_i] = 1. 
            if direction == 'gt':
                c *= -1.
                r *= -1.
            cs.append(c)
            rhs.append(r)
        all_cs.append(torch.stack(cs))
        all_rhs.append(torch.stack(rhs))
        
    lengths = [len(_) for _ in all_cs]
    if len(set(lengths)) == 1:
        return torch.stack(all_cs), torch.stack(all_rhs)
    return all_cs, all_rhs



def test_3():
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/sample.vnnlib'
    _, _, objectives = extract_instance(net_path, vnnlib_path)
    
    # for i, c_dnf in enumerate(objectives.cs):
    #     print(i)
    #     for j, c_cnf in enumerate(c_dnf):
    #         print('\t', j, c_cnf)
    #     print()
    # print(objectives.cs)
    # print(objectives.rhs)
    
    pairs = [
        [(0, 0.5, 'gt')],
        [(1, -0.5, 'gt')],
        [(1, 0.5, 'lt')],
        [(2, 0.1, 'gt')],
        [(2, 1.1, 'lt')],
        [(3, 0.2, 'gt')],
        [(3, 1.2, 'lt')],
        [(4, -0.1, 'gt')],
        [(0, 1.5, 'lt')],
        [(4, 0.31, 'lt')],
    ]
    n_outputs = 10
    
    new_cs, new_rhs = generate_simple_specs(pairs, n_outputs)
    
    print(new_cs)
    print(new_rhs)
    
    for old, new in zip(objectives.cs, new_cs):
        assert torch.equal(old, new)
        
    for old, new in zip(objectives.rhs, new_rhs):
        assert torch.equal(old, new)
    
    
@torch.no_grad
def test_4():
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    device = 'cpu'

    print('\n\nRunning test with', net_path, vnnlib_path)
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    x = torch.randn([3, *input_shape[1:]]).to(device)
    y = model(x)
    print(y.shape)
    interm_outputs = []
    
    inp = x.clone()
    for layer in list(model.modules())[1:]:
        if isinstance(layer, nn.ReLU):
            interm_outputs.append(inp)
        inp = layer(inp)
    
    assert torch.equal(inp, y)
    
    for i, v in enumerate(interm_outputs):
        print(i, v.sum())
        
    y2, interm2 = model(x, return_interm=True)
    assert torch.equal(y2, y)
            
    
    for i, v in enumerate(interm2):
        print(i, v.sum())
    
    for v1, v2 in zip(interm_outputs, interm2):
        assert torch.equal(v1, v2)
    
if __name__ == "__main__":
    test_3()