import argparse
import torch
import time

from dnn_solver.dnn_solver import DNNSolver
from dnn_solver.spec import SpecificationVNNLIB
from utils.read_vnnlib import read_vnnlib_simple
from utils.dnn_parser import DNNParser
from utils.timer import Timers
from abstract.crown import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True)
    parser.add_argument('--spec', type=str, required=True)
    parser.add_argument('--solution', action='store_true')
    parser.add_argument('--timer', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dataset', default='mnist')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--file', type=str, default='res.txt')
    args = parser.parse_args()

    device = torch.device(args.device)

    net = DNNParser.parse(args.net, args.dataset, args.device)
    spec_list = read_vnnlib_simple(args.spec, net.n_input, net.n_output)

    if args.timer:
        Timers.reset()
        Timers.tic('dnn_solver')
        
    tic = time.time()

    new_spec_list = []
    if args.dataset != 'acasxu':
        bounds = spec_list[0][0]
        for i in spec_list[0][1]:
            new_spec_list.append((bounds, [i]))
        spec_list = new_spec_list
    print(len(spec_list))

    for i, s in enumerate(spec_list):
        spec = SpecificationVNNLIB(s)
        solver = DNNSolver(net, spec)
        status = solver.solve(timeout=args.timeout)
        print('done', i, status)
        if status == 'SAT':
            break

    print(args.net, args.spec, status, time.time() - tic)
    if status=='SAT' and args.solution:
        solution = solver.get_solution()
        output = solver.net(solution)
        print('\t- lower:', spec.get_input_property()['lbs'])
        print('\t- upper:', spec.get_input_property()['ubs'])
        print('\t- solution:', solution.detach().numpy().tolist())
        print('\t- output:', output.detach().numpy().tolist())

    if args.timer:
        Timers.toc('dnn_solver')
        Timers.print_stats()

    with open(args.file, 'w') as fp:
        print(status, file=fp)