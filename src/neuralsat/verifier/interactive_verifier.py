from __future__ import annotations
import warnings
warnings.filterwarnings(action='ignore')
from beartype import beartype
import torch
import copy

from ..auto_LiRPA.utils import stop_criterion_batch_any
from ..onnx2pytorch.convert.model import ConvertModel
from ..heuristic.domains_list import DomainsList
from ..abstractor.utils import new_slopes


class InteractiveVerifier:

    "Branch-and-Bound Interactive Verifier"

    # @beartype
    def __init__(self: 'InteractiveVerifier', net: ConvertModel | torch.nn.Module , input_shape: tuple, batch: int = 1000, device: str = 'cpu') -> None:
        self.net = net # pytorch model
        self.input_shape = input_shape
        self.device = device

        # hyper parameters
        self.input_split = False
        self.batch = max(batch, 1)


    # @beartype
    def _initialize(self: 'InteractiveVerifier', objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        # initialization params
        ret = self.abstractor.initialize(objective, reference_bounds=reference_bounds)

        # check verified
        assert len(ret.output_lbs) == len(objective.cs)
        if stop_criterion_batch_any(objective.rhs.to(self.device))(ret.output_lbs.to(self.device)).all():
            return []

        # full slopes uses too much memory
        slopes = ret.slopes if self.input_split else new_slopes(ret.slopes, self.abstractor.net.final_name)

        # remaining domains
        return DomainsList(
            net=self.abstractor.net,
            objective_ids=ret.objective_ids,
            output_lbs=ret.output_lbs,
            input_lowers=ret.input_lowers,
            input_uppers=ret.input_uppers,
            lower_bounds=ret.lower_bounds,
            upper_bounds=ret.upper_bounds,
            lAs=ret.lAs,
            slopes=slopes, # pruned slopes
            histories=copy.deepcopy(ret.histories),
            cs=ret.cs,
            rhs=ret.rhs,
            input_split=self.input_split,
            preconditions=preconditions,
        )


    def decide(self):
        pick_ret = self.domains_list.pick_out(self.batch, self.device)
        decisions = self.decision(self.abstractor, pick_ret)
        return (decisions, pick_ret), None


    def init(self: 'InteractiveVerifier', objective, preconditions: dict, reference_bounds: dict | None) -> DomainsList | list:
        self._setup_restart(0, objective)
        self.domains_list = self._initialize(
            objective=objective,
            preconditions=preconditions,
            reference_bounds=reference_bounds,
        )
        return len(self.domains_list) == 0

    def step(self, action):
        obs = reward = None
        decisions, pick_ret = action
        abstraction_ret = self.abstractor.forward(decisions, pick_ret)
        self.domains_list.add(abstraction_ret, decisions)
        done = len(self.domains_list) == 0
        info = (
            self.domains_list.minimum_lowers,
            self.domains_list.visited,
        )
        return obs, reward, done, info

    from .utils import _preprocess, _init_abstractor, _setup_restart
