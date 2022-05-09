import argparse
import time

from dnn_solver.dnn_solver import DNNSolver
from dnn_solver.spec import SpecificationVNNLIB
from utils.read_vnnlib import read_vnnlib_simple
from utils.read_nnet import NetworkTorch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, required=True, help='NNET file path.')
    parser.add_argument('--spec', type=str, required=True, help='VNNLIB file path.')

    args = parser.parse_args()

    net = NetworkTorch(args.net)
    spec_list = read_vnnlib_simple(args.spec, net.input_shape[1], net.output_shape[1])

    tic = time.time()
    for i, s in enumerate(spec_list):
        spec = SpecificationVNNLIB(s)
        solver = DNNSolver(net, spec)
        status = solver.solve()
        if status:
            break

    print(args.net, args.spec, status, time.time() - tic)
    if status:
        solution = solver.get_solution()
        output = solver.dnn(solution)
        print('\t- solution:', solution)
        print('\t- output:', output.numpy().tolist())