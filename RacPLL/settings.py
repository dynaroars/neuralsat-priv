import torch

DEBUG = False

DTYPE = torch.float64

N_DECISIONS = 1
RANDOM_DECISION = False

TIGHTEN_BOUND = True
HEURISTIC_DEEPZONO = False
HEURISTIC_DEEPPOLY = not HEURISTIC_DEEPZONO
HEURISTIC_DEEPPOLY_W_ASSIGNMENT = True

HEURISTIC_RANDOMIZED_FALSIFICATION = False

HEURISTIC_DEEPPOLY_IMPLICATION = False
HEURISTIC_GUROBI_IMPLICATION = True


PARALLEL_IMPICATION = False
N_THREADS = 16
