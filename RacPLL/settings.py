import random
import torch

DEBUG = False

DTYPE = torch.float64

DECISION = 'MIN_BOUND' # 'RANDOM/MAX_BOUND/MIN_BOUND/KW/GRAD'

SEED = random.randint(0, 1000) if DECISION == 'RANDOM' else None
print('SEED:', SEED)

# N_DECISIONS = 1

TIGHTEN_BOUND = True
# HEURISTIC_DEEPZONO = False
HEURISTIC_DEEPPOLY = True
HEURISTIC_DEEPPOLY_W_ASSIGNMENT = True
# HEURISTIC_DEEPPOLY_IMPLICATION = False
HEURISTIC_DEEPPOLY_INTERVAL = 1

HEURISTIC_RANDOMIZED_FALSIFICATION = False

HEURISTIC_GUROBI_IMPLICATION = True

PARALLEL_IMPLICATION = False
N_THREADS = 16

SUPPORTED_BENCHMARKS = ['acasxu', 'cifar2020', 'mnistfc', 'oval21', 'nn4sys', 'eran', 'marabou-cifar10']
