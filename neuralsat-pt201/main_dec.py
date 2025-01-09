import traceback
import argparse
import torch
import time
import copy
import os


from util.network.read_onnx import parse_onnx, parse_pth
from util.spec.read_vnnlib import read_vnnlib

from util.misc.logger import logger, LOGGER_LEVEL
from util.misc.timer import Timers

from verifier.objective import Objective, DnfObjectives
from decomposer.dec_verifier import DecompositionalVerifier
from attacker.attacker import Attacker

from setting import Settings


def parse_pth(pth_path: str) -> tuple:
    pytorch_model = torch.load(pth_path)
    
    input_shape = (1, 3, 32, 32)
    output_shape = tuple(pytorch_model(torch.zeros(input_shape)).shape)
    is_nhwc = False
    
    # check conversion
    # correct_conversion = True
    # assert correct_conversion

    return pytorch_model, input_shape, output_shape, is_nhwc


def print_w_b(model):
    for layer in model.modules():
        if hasattr(layer, 'weight'):
            print(layer)
            print('\t[+] w:', layer.weight.data.detach().flatten())
            print('\t[+] b:', layer.bias.data.detach().flatten())
            print()

def main():
    START_TIME = time.time()

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True,
                        help="load pretrained ONNX model from this specified path.")
    parser.add_argument('--spec', type=str, required=True,
                        help="path to VNNLIB specification file.")
    parser.add_argument('--batch', type=int, default=200,
                        help="maximum number of branches to verify in each iteration")
    parser.add_argument('--timeout', type=float, default=3600,
                        help="timeout in seconds")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'],
                        help="choose device to use for verifying.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=2, 
                        help='the logger level (0: NOTSET, 1: INFO, 2: DEBUG).')
    parser.add_argument('--result_file', type=str, required=False,
                        help="file to save execution results.")
    parser.add_argument('--test', action='store_true',
                        help="test on small example with special settings.")

    args = parser.parse_args()   
    
    
    # set device
    if not torch.cuda.is_available():
        args.device = 'cpu'
        
    if args.test:
        Settings.setup_test()
    else:
        Settings.setup(args)
    
    Settings.setup_resnet_large(args)
    # Settings.setup_resnet_small(args)
    
    print(Settings)
        
    # setup timers
    if Settings.use_timer:
        Timers.reset()
        Timers.tic('Main')
        
    # set logger level
    logger.setLevel(LOGGER_LEVEL[args.verbosity])
    
    # network
    Timers.tic('Load network') if Settings.use_timer else None
    
    if args.net.endswith('.onnx'):
        model, input_shape, output_shape, is_nhwc = parse_onnx(args.net)
    elif args.net.endswith('.pth'):
        model, input_shape, output_shape, is_nhwc = parse_pth(args.net)
    else:
        raise NotImplementedError('Unsupported network type')
    
    model.eval()
    model.to(args.device)
    Timers.toc('Load network') if Settings.use_timer else None
    
    if args.verbosity:
        print(model)
        if Settings.test:
            print_w_b(model)
    
    # specification
    Timers.tic('Load specification') if Settings.use_timer else None
    vnnlibs = read_vnnlib(args.spec)
    logger.info(f'[!] Input shape: {input_shape} (is_nhwc={is_nhwc})')
    logger.info(f'[!] Output shape: {output_shape}')
    Timers.toc('Load specification') if Settings.use_timer else None
    
    # objective
    objectives = []
    for spec in vnnlibs:
        bounds = spec[0]
        for prop_i in spec[1]:
            objectives.append(Objective((bounds, prop_i)))
            
    dnf_objectives = DnfObjectives(
        objectives=objectives, 
        input_shape=input_shape, 
        is_nhwc=is_nhwc,
    )
    
    # attacker
    attacker = Attacker(
        net=model, 
        objective=dnf_objectives, 
        input_shape=input_shape, 
        device=args.device,
    )
    
    _, adv = attacker.run(timeout=10.0)
    if adv is not None:
        runtime = time.time() - START_TIME
        # export
        if args.result_file:
            os.remove(args.result_file) if os.path.exists(args.result_file) else None
            with open(args.result_file, 'w') as fp:
                print(f'sat,{runtime:.06f}', file=fp)
        print(f'sat,{runtime:.04f}')
        return
    
    
    # verifier
    verifier = DecompositionalVerifier(
        net=model,
        input_shape=input_shape,
        min_layer=Settings.min_layer,
        device=args.device,
    )    
    
    Timers.tic('Verify') if Settings.use_timer else None
    timeout = args.timeout - (time.time() - START_TIME)
    
    
    # verify
    try:
        if Settings.use_decompose:
            status = verifier.decompositional_verify(
                objectives=copy.deepcopy(dnf_objectives), 
                timeout=timeout, 
                batch=args.batch,
            )
        else:
            status = verifier.original_verify(
                objectives=copy.deepcopy(dnf_objectives), 
                timeout=timeout, 
                batch=args.batch,
            )
    except:
        print(traceback.format_exc())
        status = 'unknown'
    
    runtime = time.time() - START_TIME
    Timers.toc('Verify') if Settings.use_timer else None
    
    # output
    logger.info(f'[!] Iterations: {verifier.iteration}')
    logger.info(f'[!] Result: {status}')
    logger.info(f'[!] Runtime: {runtime:.04f}')
        
    # export
    if args.result_file:
        os.remove(args.result_file) if os.path.exists(args.result_file) else None
        with open(args.result_file, 'w') as fp:
            print(f'{status},{runtime:.06f}', file=fp)

    if Settings.use_timer:
        Timers.toc('Main')
        Timers.print_stats()
        
    print(f'{status},{runtime:.04f}')
        
if __name__ == '__main__':
    main()