from utils.read_vnnlib import read_vnnlib_simple

import itertools
import random
import torch



def split_bounds(bounds, steps=3):
    lower = bounds['lbs']
    upper = bounds['ubs']

    bs = [(l, u) for l, u in zip(lower, upper)]
    bs = [torch.linspace(b[0], b[1], steps=steps) for b in bs]
    bs = [[torch.Tensor([b[i], b[i+1]]) for i in range(b.shape[0] - 1)] for b in bs]
    bs = itertools.product(*bs)
    splits = [{'lbs': torch.Tensor([_[0] for _ in b]),
               'ubs': torch.Tensor([_[1] for _ in b])} for b in bs]
    random.shuffle(splits)
    return splits


class SpecificationVNNLIB:

    def __init__(self, spec):

        self.bounds, self.mat = spec

    def get_input_property(self):
        return {
            'lbs': [b[0] for b in self.bounds],
            'ubs': [b[1] for b in self.bounds]
        }

    def get_output_property(self, output):
        dnf =  []
        for lhs, rhs in self.mat:
            cnf = []
            for l, r in zip(lhs, rhs):
                cnf.append(sum(l * output) <= r)
            dnf.append(cnf)
        return dnf


    def check_output_reachability(self, lbs, ubs):
        dnf =  []
        for lhs, rhs in self.mat:
            lhs = torch.tensor(lhs, dtype=torch.float32)
            rhs = torch.tensor(rhs, dtype=torch.float32)

            cnf = []
            for l, r in zip(lhs, rhs):
                cnf.append(sum((l > 0) * lbs) <=  sum((l < 0) * ubs))
            dnf.append(all(cnf))
        return any(dnf)

    def check_solution(self, output):
        for lhs, rhs in self.mat:
            lhs = torch.tensor(lhs, dtype=torch.float32)
            rhs = torch.tensor(rhs, dtype=torch.float32)
            vec = lhs @ output
            if torch.all(vec <= rhs):
                return True
        return False