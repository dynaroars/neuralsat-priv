import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import argparse
import torch
import sys
import os


from util.network.read_onnx import parse_onnx
from util.spec.read_vnnlib import read_vnnlib
from util.misc.logger import logger

from verifier.objective import Objective, DnfObjectives
from attacker.attacker import Attacker
from train.models.vit.vit import *


def falsify(net_path, spec_path, device):
    model, input_shape, output_shape, is_nhwc = parse_onnx(net_path)
    model.to(device)
    
    vnnlibs = read_vnnlib(spec_path)
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
            
    objectives = DnfObjectives(
        objectives=objectives, 
        input_shape=input_shape, 
        is_nhwc=is_nhwc,
    )
    
    # falsifier
    attacker = Attacker(
        net=model, 
        objective=objectives, 
        input_shape=input_shape, 
        device=device,
    )
    
    # attack
    is_attacked, adv = attacker.run(timeout=20.0)
    
    return is_attacked
    
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--benchmark_dir', default='example/generated_benchmark/')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--eps', type=float, default=0.02)
    parser.add_argument('--timeout', type=float, default=1000.0)

    args = parser.parse_args()
    # args.device = torch.device(args.device)
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = os.path.join(args.benchmark_dir, args.model_type, args.model_name, f'eps_{args.eps:.06f}')
    in_csv_path = os.path.join(output_dir, 'instances.csv')
    out_csv_path = os.path.join(output_dir, 'unattacked_instances.csv')
    
    total = 0
    skip = 0
    with open(out_csv_path, 'w') as fp:
        pbar = tqdm(open(in_csv_path).read().strip().split('\n'), desc=f'Attacking instances for "{args.model_name}" eps={args.eps}')
        for line in pbar:
            net_path, spec_path, _ = line.split(',')
            data = {
                'net_path': os.path.join(output_dir, net_path),
                'spec_path': os.path.join(output_dir, spec_path),
                'device': args.device,
            }
            is_attacked = falsify(**data)
            # print(f'{is_attacked=}')
            if not is_attacked:
                print(line, file=fp)
                total += 1
                pbar.set_postfix(total=total, skip=skip)
            else:
                skip += 1
                
                
            # if is_attacked:
            #     print(line)
            #     print(data)
            #     raise
    
if __name__ == "__main__":
    main()