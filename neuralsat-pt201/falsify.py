import argparse
import torch
import time
import os

from util.network.read_onnx import parse_onnx
from util.spec.read_vnnlib import read_vnnlib
from util.misc.logger import logger

from verifier.objective import Objective, DnfObjectives
from attacker.attacker import Attacker

if __name__ == '__main__':
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--timeout', type=float, default=5.0,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    
    args = parser.parse_args()   

    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    # network
    model, input_shape, output_shape, is_nhwc = parse_onnx(args.net)
    model.to(args.device)
    print(model)
    logger.info(f'[!] Input shape: {input_shape} (is_nhwc={is_nhwc})')
    logger.info(f'[!] Output shape: {output_shape}')
    
    # specification
    vnnlibs = read_vnnlib(args.spec)
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
        device=args.device,
    )
    
    # attack
    status = 'unknown'
    is_attacked, adv = attacker.run(timeout=args.timeout)
    runtime = time.time() - START_TIME
    if is_attacked:
        status = 'sat'
        
    print(f'{status},{runtime:.04f}')
        