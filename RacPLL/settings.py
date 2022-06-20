import random
import torch

DEBUG = False

DTYPE = torch.float64

DECISION = 'MIN_BOUND' # 'RANDOM/MAX_BOUND/MIN_BOUND/KW/GRAD'

SEED = random.randint(0, 10000) if DECISION == 'RANDOM' else None
print('SEED:', SEED)

HEURISTIC_DEEPZONO = False
HEURISTIC_DEEPPOLY = True
HEURISTIC_DEEPPOLY_W_ASSIGNMENT = True
# HEURISTIC_DEEPPOLY_IMPLICATION = False
HEURISTIC_DEEPPOLY_INTERVAL = 1

HEURISTIC_RANDOMIZED_FALSIFICATION = True

HEURISTIC_GUROBI_IMPLICATION = True

PARALLEL_IMPLICATION = True
N_THREADS = 16

BENCHMARKS = ['acasxu', 'cifar2020', 'mnistfc']#, 'oval21', 'nn4sys', 'eran', 'marabou-cifar10']
