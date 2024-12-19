import warnings
warnings.filterwarnings(action='ignore')

from beartype import beartype
import traceback
import logging
import typing
import torch
import copy
import math
import os

from auto_LiRPA.utils import stop_criterion_batch_any
from auto_LiRPA import BoundedModule

from util.misc.result import AbstractResults, CoefficientMatrix
from onnx2pytorch.convert.model import ConvertModel
from util.misc.logger import logger
from abstractor.params import *


class NetworkAbstractor:
    
    "Over-approximation method alpha-beta-CROWN"

    @beartype
    def __init__(self: 'NetworkAbstractor', pytorch_model: ConvertModel | torch.nn.Module, 
                 input_shape: tuple, method: str, input_split: bool = False, device: str = 'cpu') -> None:

        self.pytorch_model = copy.deepcopy(pytorch_model)
        self.device = device
        self.input_shape = input_shape
        
        # search domain
        self.input_split = input_split
        
        # computation algorithm
        self.method = method
        
        # debug
        self.iteration = 0
        
    @beartype
    @property
    def split_points(self):
        if not hasattr(self, '_split_points'):
            self._split_points = [self.net.split_activations[k.name][0][0].get_split_point() for k in self.net.split_nodes]
        return self._split_points
        
    @beartype
    def setup(self: 'NetworkAbstractor', objective: typing.Any, extra_opts: dict = {}) -> None:
        if self.select_params(objective, extra_opts=extra_opts):
            logger.info(f'[setup] Initialized abstractor: {self.mode=}, {self.method=}, {self.input_split=}, {Settings.backward_batch_size=} {extra_opts=}')
            return None
        
        # FIXME: try special settings for ViT
        new_extra_opts = copy.deepcopy(extra_opts)
        new_extra_opts.update({'sparse_intermediate_bounds': False})
        if self.select_params(objective, extra_opts=new_extra_opts):
            logger.info(f'[setup] Initialized abstractor: {self.mode=}, {self.method=}, {self.input_split=}, {new_extra_opts=}')
            return None
            
        # FIXME: try smaller backward batch size
        Settings.backward_batch_size = 512
        while Settings.backward_batch_size >= 1:
            if self.select_params(objective, extra_opts=extra_opts):
                logger.info(f'[setup] Initialized abstractor: {self.mode=}, {self.method=}, {self.input_split=}, {Settings.backward_batch_size=} {extra_opts=}')
                return None 
            Settings.backward_batch_size = Settings.backward_batch_size // 2

        logger.info('[setup] Initialization failed')
        raise
            
    @beartype
    def select_params(self: 'NetworkAbstractor', objective: typing.Any, extra_opts: dict = {}) -> bool:
        params = [
            ['patches', self.method], # default
            ['matrix', self.method],
        ]
        if self.input_split and (self.method != 'backward'):
            params += [        
                ['patches', 'backward'],
                ['matrix', 'backward'],
            ]
        
        for mode, method in params:
            logger.debug(f'[select_params] Try {mode=}, {method=}, {self.input_split=} {extra_opts=}')
            self._init_module(mode=mode, objective=objective, extra_opts=extra_opts)
            if self._check_module(method=method, objective=objective):
                self.mode = mode
                self.method = method
                return True
            
        return False
            
    @beartype
    def _init_module(self: 'NetworkAbstractor', mode: str, objective: typing.Any, extra_opts: dict = {}) -> None:
        bound_opts = {'conv_mode': mode, 'verbosity': 0, **extra_opts}
        logger.debug(f'[_init_module] {bound_opts=}')
        self.net = BoundedModule(
            model=self.pytorch_model, 
            global_input=torch.zeros(self.input_shape, device=self.device),
            bound_opts=bound_opts,
            device=self.device,
            verbose=False,
        )
        self.net.eval()
        self.net.get_split_nodes()
        
        # check conversion correctness
        if objective:
            dummy = objective.lower_bounds[0].clone().view(self.input_shape).to(self.device)
        else:
            logger.debug(f'[_init_module] Use random dummy input for checking correctness')
            dummy = torch.randn(self.input_shape, device=self.device) 
            
        # FIXME: remove
        if 0:
            try:
                assert torch.allclose(self.pytorch_model(dummy), self.net(dummy), atol=1e-4, rtol=1e-4)
            except:
                print('[!] Conversion error')
                raise ValueError(f'torch allclose failed: {torch.norm(self.pytorch_model(dummy) - self.net(dummy))}')
        
        
    @beartype
    def _check_module(self: 'NetworkAbstractor', method: str, objective: typing.Any) -> bool:
        # at least can run with batch=1
        if objective:
            x_L = objective.lower_bounds[0].view(self.input_shape)
            x_U = objective.upper_bounds[0].view(self.input_shape)
        else:
            logger.debug(f'[_check_module] Use random dummy input for checking correctness')
            x_L = torch.randn(self.input_shape, device=self.device) 
            x_U = x_L + 0.1
        
        x = self.new_input(x_L=x_L, x_U=x_U)
        
        # forward to save architectural information
        self.net(x) 
    
        if math.prod(self.input_shape) >= 100000:
            return True
        
        return True # TODO: remove
    
        try:
            self.net.set_bound_opts(get_check_abstractor_params())
            self.net.init_alpha(x=(x,)) if method == 'crown-optimized' else None
            lb, _ = self.net.compute_bounds(x=(x,), method=method) # FIXME: this uses a lot of RAM
            assert not torch.isnan(lb).any()
        except KeyboardInterrupt:
            exit()
        except SystemExit:
            exit()
        except:
            # NOTE: MUST COMMENT this line, only UNCOMMENT for debugging 
            if os.environ.get("NEURALSAT_DEBUG"):
                raise 
            
            if logger.level <= logging.DEBUG:
                traceback.print_exc()
            else:
                logger.info(f'[!] Error when trying method="{method}", backward_batch_size={Settings.backward_batch_size}')
            return False
        else:
            return True
        

    @beartype
    def initialize(self: 'NetworkAbstractor', objective: typing.Any, 
                   share_alphas: list = [], reference_bounds: dict | None = None, 
                   short_cut: bool = False) -> AbstractResults:
        objective.cs = objective.cs.to(self.device)
        objective.rhs = objective.rhs.to(self.device)
        
        # input property
        input_lowers = objective.lower_bounds.view(-1, *self.input_shape[1:]).to(self.device)
        input_uppers = objective.upper_bounds.view(-1, *self.input_shape[1:]).to(self.device)
       
        # stop function used when optimizing abstraction
        stop_criterion_func = stop_criterion_batch_any(objective.rhs)
        
        # create input
        x = self.new_input(x_L=input_lowers, x_U=input_uppers)
        
        # update initial reference bounds for later use
        self.init_reference_bounds = reference_bounds
        
        # get split nodes
        self.net.get_split_nodes()
        
        if self.method not in ['crown-optimized']:
            with torch.no_grad():
                lb, _ = self.net.compute_bounds(
                    x=(x,), 
                    C=objective.cs, 
                    method=self.method, 
                    reference_bounds=reference_bounds,
                )
            logger.info(f'Initial bounds (fisrt 10): {lb.detach().cpu().flatten()[:10]}')
            if stop_criterion_func(lb).all().item():
                return AbstractResults(**{'output_lbs': lb})
            
            if short_cut:
                return AbstractResults(**{
                    'objective_ids': getattr(objective, 'ids', None),
                    'output_lbs': lb, 
                    'lAs': self.get_lAs(), 
                    'slopes': self.get_slope(), 
                    'cs': objective.cs,
                    'rhs': objective.rhs,
                    'input_lowers': input_lowers,
                    'input_uppers': input_uppers,
                })
                
                
            # reorganize tensors
            # FIXME: AssertionError: Hidden lower: [1, 1, 1, 1, 9]
            with torch.no_grad():
                lower_bounds, upper_bounds = self.get_hidden_bounds(lb)
                
            return AbstractResults(**{
                'objective_ids': getattr(objective, 'ids', None),
                'output_lbs': lb, 
                'lAs': self.get_lAs(), 
                'lower_bounds': lower_bounds, 
                'upper_bounds': upper_bounds, 
                'slopes': self.get_slope(), 
                'histories': {_.name: ([], [], []) for _ in self.net.split_nodes}, 
                'cs': objective.cs,
                'rhs': objective.rhs,
                'input_lowers': input_lowers,
                'input_uppers': input_uppers,
            })
            
        # setup optimization parameters
        self.net.set_bound_opts(get_initialize_opt_params(stop_criterion_func))

        # initial bounds
        lb_init, _, aux_reference_bounds = self.net.init_alpha(
            x=(x,), 
            share_alphas=share_alphas, 
            c=objective.cs, 
            bound_upper=False,
        )
        print(f'[Init alpha] {x.shape=} {share_alphas=} {objective.cs.shape=}',)
        print(f'[Init alpha] {lb_init.flatten()=}\n\n')
        logger.info(f'Initial bounds (fisrt 10): {lb_init.detach().cpu().flatten()[:10]}')
    
        if stop_criterion_func(lb_init).all().item():
            return AbstractResults(**{'output_lbs': lb_init})

        # self.update_refined_beta(init_betas, batch=len(objective.cs))
        lb, _ = self.net.compute_bounds(
            x=(x,), 
            C=objective.cs, 
            method='crown-optimized',
            aux_reference_bounds=aux_reference_bounds, 
            reference_bounds=reference_bounds,
        )
        print(f'[Optimized alpha] {lb.flatten()=}',)
        logger.info(f'Initial optimized bounds (fisrt 10): {lb.detach().cpu().flatten()[:10]}')
        if stop_criterion_func(lb).all().item():
            return AbstractResults(**{'output_lbs': lb})
        
        # reorganize tensors
        with torch.no_grad():
            lower_bounds, upper_bounds = self.get_hidden_bounds(lb)

        return AbstractResults(**{
            'objective_ids': objective.ids,
            'output_lbs': lower_bounds[self.net.final_name], 
            'lAs': self.get_lAs(), 
            'lower_bounds': lower_bounds, 
            'upper_bounds': upper_bounds, 
            'slopes': self.get_slope(), 
            'histories': {_.name: ([], [], []) for _ in self.net.split_nodes}, 
            'cs': objective.cs,
            'rhs': objective.rhs,
            'input_lowers': input_lowers,
            'input_uppers': input_uppers,
        })
            
    
    @beartype
    def _forward_hidden(self: 'NetworkAbstractor', domain_params: AbstractResults, decisions: list, simplify: bool) -> AbstractResults:
        assert len(decisions) == len(domain_params.cs) == len(domain_params.rhs) == \
               len(domain_params.input_lowers) == len(domain_params.input_uppers), \
               print(f'len(decisions)={len(decisions)}, len(domain_params.input_lowers)={len(domain_params.input_lowers)}')
            
        batch = len(decisions)
        assert batch > 0
        
        # 2 * batch
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_input_lowers = torch.cat([domain_params.input_lowers, domain_params.input_lowers], dim=0) # TODO: torch compile
        double_input_uppers = torch.cat([domain_params.input_uppers, domain_params.input_uppers], dim=0) # TODO: torch compile
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(double_input_lowers <= double_input_uppers)
        
        # update hidden bounds with new decisions (perform splitting)
        new_intermediate_layer_bounds = self.hidden_split_idx(
            lower_bounds=domain_params.lower_bounds, 
            upper_bounds=domain_params.upper_bounds, 
            decisions=decisions
        )
        
        # create new inputs
        new_x = self.new_input(x_L=double_input_lowers, x_U=double_input_uppers)
        
        # update slopes
        if len(domain_params.slopes) > 0: 
            self.set_slope(domain_params.slopes)
            
        # simplify for decision heuristics
        if simplify:
            # setup optimization parameters
            self.net.set_bound_opts(get_branching_opt_params())
            
            # compute outputs
            with torch.no_grad():
                double_output_lbs, _, = self.net.compute_bounds(
                    x=(new_x,), 
                    C=double_cs, 
                    method='backward', 
                    reuse_alpha=self.method == 'crown-optimized',
                    interm_bounds=new_intermediate_layer_bounds
                )
            return AbstractResults(**{'output_lbs': double_output_lbs})

        # 2 * batch
        assert len(decisions) == len(domain_params.objective_ids)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        double_objective_ids = torch.cat([domain_params.objective_ids, domain_params.objective_ids], dim=0)
        double_sat_solvers = domain_params.sat_solvers * 2 if domain_params.sat_solvers is not None else None
         
        # update new decisions
        double_histories = self.update_histories(histories=domain_params.histories, decisions=decisions)
        double_betas = domain_params.betas * 2
        num_splits = self.set_beta(betas=double_betas, histories=double_histories)
        
        # setup optimization parameters
        self.net.set_bound_opts(get_beta_opt_params(stop_criterion_batch_any(double_rhs)))
        
        # compute outputs
        double_ref_output_lbs = torch.cat([domain_params.output_lbs, domain_params.output_lbs], dim=0) # TODO: torch compile
        reference_bounds = {self.net.final_name: [double_ref_output_lbs, double_ref_output_lbs + torch.inf]}
        double_output_lbs, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            decision_thresh=double_rhs,
            interm_bounds=new_intermediate_layer_bounds,
            reference_bounds=reference_bounds,
        )

        # reorganize output
        with torch.no_grad():
            # lAs
            double_lAs = self.get_lAs()
            # outputs
            double_output_lbs = double_output_lbs.detach().to(device='cpu')
            # slopes
            double_slopes = self.get_slope() if len(domain_params.slopes) > 0 else {}
            # betas
            double_betas = self.get_beta(num_splits)
            # hidden bounds
            double_lower_bounds, double_upper_bounds = self.get_hidden_bounds(double_output_lbs)
            
        assert all([_.shape[0] == 2 * batch for _ in double_lower_bounds.values()]), print([_.shape for _ in double_lower_bounds.values()])
        assert all([_.shape[0] == 2 * batch for _ in double_upper_bounds.values()]), print([_.shape for _ in double_upper_bounds.values()])
        assert all([_.shape[0] == 2 * batch for _ in double_lAs.values()]), print([_.shape for _ in double_lAs.values()])
        assert len(double_histories) == len(double_betas) == 2 * batch
            
        return AbstractResults(**{
            'objective_ids': double_objective_ids,
            'output_lbs': double_lower_bounds[self.net.final_name], 
            'input_lowers': double_input_lowers, 
            'input_uppers': double_input_uppers,
            'lAs': double_lAs, 
            'lower_bounds': double_lower_bounds, 
            'upper_bounds': double_upper_bounds, 
            'slopes': double_slopes, 
            'betas': double_betas, 
            'histories': double_histories,
            'cs': double_cs,
            'rhs': double_rhs,
            'sat_solvers': double_sat_solvers,
        })
        
        
    @beartype
    def _forward_input(self: 'NetworkAbstractor', domain_params: AbstractResults, decisions: torch.Tensor, simplify: bool) -> AbstractResults:
        assert len(decisions) == len(domain_params.cs) == len(domain_params.rhs) == \
               len(domain_params.input_lowers) == len(domain_params.input_uppers) == len(domain_params.objective_ids)
               
        batch = len(decisions)
        assert batch > 0
        
        # splitting input by decisions (perform splitting)
        new_input_lowers, new_input_uppers = self.input_split_idx(
            input_lowers=domain_params.input_lowers, 
            input_uppers=domain_params.input_uppers, 
            split_idx=decisions,
        )
        
        # create new inputs
        new_x = self.new_input(x_L=new_input_lowers, x_U=new_input_uppers)
        
        # 2 * batch
        double_objective_ids = torch.cat([domain_params.objective_ids, domain_params.objective_ids], dim=0)
        double_cs = torch.cat([domain_params.cs, domain_params.cs], dim=0)
        double_rhs = torch.cat([domain_params.rhs, domain_params.rhs], dim=0)
        
        # set slope again since batch might change
        if len(domain_params.slopes) > 0: 
            self.set_slope(domain_params.slopes)
        
        # set optimization parameters
        self.net.set_bound_opts(get_input_opt_params(stop_criterion_batch_any(double_rhs)))
        
        double_output_lbs, _ = self.net.compute_bounds(
            x=(new_x,), 
            C=double_cs, 
            method=self.method,
            decision_thresh=double_rhs,
            reference_bounds=self.init_reference_bounds,
        )

        with torch.no_grad():
            # slopes
            double_slopes = self.get_slope() if len(domain_params.slopes) > 0 else {}
            double_lAs = self.get_lAs()

        return AbstractResults(**{
            'objective_ids': double_objective_ids,
            'output_lbs': double_output_lbs, 
            'input_lowers': new_input_lowers, 
            'input_uppers': new_input_uppers,
            'slopes': double_slopes, 
            'lAs': double_lAs, 
            'cs': double_cs, 
            'rhs': double_rhs, 
        })
        
        
    @beartype
    def forward(self: 'NetworkAbstractor', decisions: list | torch.Tensor, domain_params: AbstractResults) -> AbstractResults:
        self.iteration += 1
        forward_func = self._forward_input if self.input_split else self._forward_hidden
        return forward_func(domain_params=domain_params, decisions=decisions, simplify=False)

    
    # TODO: experimental function
    def compute_bounds(self, input_lowers, input_uppers, method, cs=None, rhs=None, reference_bounds=None, reuse_alpha=False):
        assert method in ['backward', 'crown-optimized']
        if os.environ.get('NEURALSAT_ASSERT'):
            assert not torch.equal(input_lowers, input_uppers)
        
        # perturbed input
        x = self.new_input(x_L=input_lowers, x_U=input_uppers)

        # get split nodes
        self.net.get_split_nodes()
    
        # if Settings.share_alphas:
        #     print(f'[!] Using {Settings.share_alphas=} will lose precision.')
        coeffs = None
        
        # if reuse_alpha:
        #     with torch.no_grad():
        #         lb, ub, = self.net.compute_bounds(
        #             x=(x,), 
        #             C=cs, 
        #             method='backward', 
        #             reuse_alpha=reuse_alpha,
        #             # interm_bounds=new_intermediate_layer_bounds
        #         )
        #     return (lb, ub), coeffs

        # setup options for optimization mode
        self.net.set_bound_opts(get_initialize_opt_params(lambda x: False))
        
        # backward mode
        lb, ub, aux_reference_bounds = self.net.init_alpha(
            x=(x,), 
            c=cs, 
            share_alphas=Settings.share_alphas, 
            bound_upper=True,
        )
        assert torch.all(lb <= ub + 1e-6), f'{(lb > ub).sum()} {lb[lb > ub]} {ub[lb > ub]}'
        # print(f'Inititial bounds with {method=}:', lb.detach().cpu())
        
        if method == 'backward':
            lA, uA, lbias, ubias = self.get_input_A(self.device)
            coeffs = CoefficientMatrix(lA=lA, uA=uA, lbias=lbias, ubias=ubias)
            return (lb, ub), coeffs

        # lower bound
        lb, _ = self.net.compute_bounds(
            x=(x,), 
            C=cs,
            method=method,
            aux_reference_bounds=aux_reference_bounds, 
            reference_bounds=reference_bounds,
            bound_lower=True,
            bound_upper=False,
        )
        lA, _, lbias, _ = self.get_input_A(self.device)
        # print(f'Optimized bounds with {method=}:', lb.detach().cpu())
        
        # upper bound
        _, ub = self.net.compute_bounds(
            x=(x,), 
            C=cs,
            method=method,
            aux_reference_bounds=aux_reference_bounds, 
            reference_bounds=reference_bounds,
            bound_lower=False,
            bound_upper=True,
        )
        _, uA, _, ubias = self.get_input_A(self.device)
        coeffs = CoefficientMatrix(lA=lA, uA=uA, lbias=lbias, ubias=ubias)
        assert torch.all(lb <= ub + 1e-6), f'{(lb > ub).sum()} {lb[lb > ub]} {ub[lb > ub]}'
        
        return (lb, ub), coeffs
        
        
    def __repr__(self):
        return f'{self.__class__.__name__}({self.mode}, {self.method})'
        
        
    from .utils import (
        new_input,
        get_slope, set_slope,
        get_beta, set_beta, reset_beta, update_refined_beta,
        get_hidden_bounds,
        get_lAs, get_input_A,
        update_histories,
        hidden_split_idx, input_split_idx,
        build_lp_solver, solve_full_assignment,
        compute_stability,
    )
