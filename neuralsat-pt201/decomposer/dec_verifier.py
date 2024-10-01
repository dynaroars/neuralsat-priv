from collections import namedtuple
from beartype import beartype
import torch.nn as nn
import traceback
import logging
import random
import typing
import torch
import time
import copy
import tqdm
import sys
import os

from util.network.read_onnx import parse_onnx, decompose_pytorch
from util.spec.write_vnnlib import write_vnnlib

from util.misc.torch_cuda_memory import is_cuda_out_of_memory, gc_cuda
from util.misc.result import ReturnStatus, CoefficientMatrix
from util.misc.logger import logger

from attacker.pgd_attack.general import attack as pgd_attack
from tightener.utils import optimize_dnn, verify_dnf_pairs
from verifier.objective import DnfObjectives
from verifier.verifier import Verifier

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedModule, BoundedTensor

from setting import Settings

from train.models.vit.vit import *
from train.models.resnet.resnet import *



SubNetworks = namedtuple('SubNetworks', ['network', 'input_shape', 'output_shape'])
InOutBounds = namedtuple('InputOutputBounds', ['under_input', 'under_output', 'over_input', 'over_output'], defaults=(None,) * 4)

class PytorchWrapper(nn.Module):

    def __init__(self, module_lists):
        super(PytorchWrapper, self).__init__()
        self.layers = module_lists
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    

class DecompositionalVerifier:
    
    @beartype
    def __init__(self, net: nn.Module, input_shape: tuple, min_layer: int, device: str = 'cpu') -> None:
        self.net = net.to(device) # pytorch model
        self.device = device
        self.input_shape = input_shape
        self.min_layer = min_layer
        
    @beartype
    def reset(self) -> None:
        if not hasattr(self, 'input_output_bounds'):
            self.input_output_bounds = {k: None for k in range(len(self.sub_networks))}
        
        if not hasattr(self, 'extras'):
            self.extras = {0: None}
        
        self.tightening_candidates = {}
        
    @beartype
    def decompose_network_auto(self, net: nn.Module, input_shape: tuple, min_layer: int) -> None:
        self.sub_networks = {}
        
        tmp_net = net
        tmp_shape = input_shape
        count = 0
        while True:
            prefix, suffix = self._split_network(tmp_net, tmp_shape, min_layer=min_layer)
            self.sub_networks[count] = SubNetworks(*prefix)
            if suffix is None:
                break 
            tmp_net, tmp_shape, _ = suffix
            count += 1
            # FIXME: support more than 2
            min_layer = 10000
            print(prefix)
            exit()
        
        # FIXME: support more than 2
        assert len(self.sub_networks) <= 2
        
        # check correctness
        dummy = torch.randn(2, *input_shape[1:], device=self.device)
        output_1 = net(dummy)
        output_2 = dummy.clone()
        for idx in range(len(self.sub_networks)):
            output_2 = self.sub_networks[idx].network(output_2)
        assert torch.allclose(output_1, output_2)
        print(f'[+] Passed {len(self.sub_networks)=}')

    @beartype
    def decompose_network(self, net: nn.Module, input_shape: tuple, min_layer: int) -> None:
        self.sub_networks = {}
        if not (hasattr(net, 'layers') and isinstance(net.layers, nn.ModuleList)):
            return self.decompose_network_auto(net, input_shape, min_layer)
        
        assert isinstance(net.layers, nn.ModuleList)
        
        in_shape = input_shape
        for count, idx in enumerate(range(0, len(net.layers), min_layer)):
            subnet = PytorchWrapper(net.layers[idx:idx+min_layer])
            subnet.eval()
            out_shape = subnet(torch.randn(in_shape, device=self.device)).size()
            self.sub_networks[count] = SubNetworks(subnet, in_shape, out_shape)
            in_shape = out_shape
            
        # checking correctness
        dummy = torch.randn(2, *input_shape[1:], device=self.device)
        y1 = net(dummy)
        
        y2 = dummy
        for i in range(len(self.sub_networks)):
            y2 = self.sub_networks[i].network(y2)
            
        assert torch.equal(y1, y2)
        print(f'[+] Passed decomposing network: {len(self.sub_networks)=}')
        
        # for idx in range(len(self.sub_networks)):
        #     print(self.sub_networks[idx].network)
        #     subnet_input_shape = self.sub_networks[idx].input_shape
        #     subnet_output_shape = self.sub_networks[idx].output_shape
        #     print(f'{idx=} {subnet_input_shape=} {subnet_output_shape=}')
        #     print()
        
        

    @beartype
    def _split_network(self, net: nn.Module, input_shape: tuple, min_layer: int) -> tuple[tuple, tuple | None]:
        split_idx = min_layer
        while True:
            prefix_onnx_byte, suffix_onnx_byte = decompose_pytorch(net, input_shape, split_idx + 1)
            net = net.to(self.device)
            if (prefix_onnx_byte is None) or (suffix_onnx_byte is None):
                return (net, input_shape, None), None
            assert prefix_onnx_byte is not None

            # parse subnets
            try:
                prefix, prefix_input_shape, prefix_output_shape, _ = parse_onnx(prefix_onnx_byte)
                suffix, suffix_input_shape, suffix_output_shape, _ = parse_onnx(suffix_onnx_byte)
            except:
                print(f'Failed to split at {split_idx=}')
                # traceback.print_exc()
                split_idx += 1
                continue

            # move to device
            print(f'Succeeded to split at {split_idx=}')
            prefix = prefix.to(self.device)
            suffix = suffix.to(self.device)
            print(f'{prefix=}')
            print(f'{suffix=}')
                
            exit()
            # check correctness
            dummy = torch.randn(2, *prefix_input_shape[1:], device=self.device) # try batch=2
            if torch.allclose(net(dummy), suffix(prefix(dummy))):
                return (prefix.to(self.device), prefix_input_shape, prefix_output_shape), (suffix.to(self.device), suffix_input_shape, suffix_output_shape)
            
    @beartype
    def _setup_subnet_verifier(self, subnet_idx: int, objective: typing.Any | None = None, batch: int=500) -> Verifier:
        subnet_params = self.sub_networks[subnet_idx]
        # network
        if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
            network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
        else:
            # network = torch.nn.Sequential(torch.nn.Identity(), subnet_params.network)
            network = subnet_params.network
        print(network)
        
        verifier = Verifier(
            net=network,
            input_shape=subnet_params.input_shape,
            batch=batch,
            device=self.device,
        )
        verifier._setup_restart_naive(0, objective)
        verifier.abstractor.extras = self.extras.get(subnet_idx, None)
        
        return verifier

        
    def compute_bounds(self, abstractor, input_lowers, input_uppers, cs, method, output_sequential=False, output_batch=10):
        
        diff = (input_uppers - input_lowers).clone()
        eps = diff.max().item()
        
        if not output_sequential:
            print(f'{cs=} {cs.shape=} {eps=}')
            return abstractor.compute_bounds(
                input_lowers=input_lowers,
                input_uppers=input_uppers,
                cs=cs,
                method=method,
            )
            
        assert len(input_lowers) == 1, f'Only support batch=1: {len(input_lowers)=}'
        n_outputs = abstractor.net(input_lowers).flatten(1).shape[1]
        
        # print(f'{n_outputs=}')
        output_lowers, output_uppers = [], []
        output_coeffs = []
        
        pbar = tqdm.tqdm(range(0, n_outputs, output_batch), desc=f'Compute bounds: {output_batch=} {method=} {eps=:.06f}')
        for i in pbar:
            indices = torch.arange(i, min(i+output_batch, n_outputs))
            ci = torch.nn.functional.one_hot(indices, num_classes=n_outputs)[None].to(input_lowers)
            (lb, ub), coeffs = abstractor.compute_bounds(
                input_lowers=input_lowers,
                input_uppers=input_uppers,
                cs=ci,
                method=method,
            )
            # print(ci.shape, lb.shape)
            output_lowers.append(lb)
            output_uppers.append(ub)
            # print(f'{coeffs.lA.shape=}')
            if coeffs:
                output_coeffs.append(coeffs)
                raise # TODO: disabling coeffs might save memory
            
        output_lowers = torch.cat(output_lowers, dim=-1)
        output_uppers = torch.cat(output_uppers, dim=-1)
        
        
        if output_coeffs:
            output_coeffs = CoefficientMatrix(
                lA=torch.cat([c.lA for c in output_coeffs], dim=0),
                uA=torch.cat([c.uA for c in output_coeffs], dim=0),
                lbias=torch.cat([c.lbias for c in output_coeffs], dim=-1),
                ubias=torch.cat([c.ubias for c in output_coeffs], dim=-1),
            )
            raise # TODO: disabling coeffs might save memory
        
        return (output_lowers, output_uppers), output_coeffs
        

    @beartype
    def _init_interm_bounds(self, objective: typing.Any, use_extra: bool = True, method='crown-optimized') -> tuple:
        # method = 'backward'
        print(f'Init interm bounds {method=}')
        # input 
        input_shape = self.sub_networks[0].input_shape
        input_lb_0 = objective.lower_bounds.view(input_shape).to(self.device)
        input_ub_0 = objective.upper_bounds.view(input_shape).to(self.device)
        if self.input_output_bounds[0] is None:
            self.input_output_bounds[0] = InOutBounds(over_input=(input_lb_0.clone(), input_ub_0.clone()))
        
        n_subnets = len(self.sub_networks)
        
        for idx in range(n_subnets):
            # if idx <=1:
            #     method = 'backward'
            # else:
            #     method = 'crown-optimized'
                
            if (idx != n_subnets-1) and (self.input_output_bounds[idx].over_output is not None):
                # no need to re-init for different objectives, only need to update last over_output due to cs
                # TODO: recheck input property
                print(f'Reuse computed bounds subnet {idx=}')
                continue
            print(f'Processing subnet {idx+1}/{n_subnets}')
            verifier = self._setup_subnet_verifier(idx)
            print(f'{verifier.abstractor.net=}')
            print(verifier.net)
            print(self.sub_networks[idx].input_shape)
            print(self.sub_networks[idx].output_shape)
            
            (output_lb, output_ub), output_coeffs = self.compute_bounds(
                abstractor=verifier.abstractor,
                input_lowers=self.input_output_bounds[idx].over_input[0],
                input_uppers=self.input_output_bounds[idx].over_input[1],
                cs=objective.cs if idx==n_subnets-1 else None,
                method=method,
                output_sequential=idx!=n_subnets-1,
                output_batch=50,
            )
            # exit()
            
            # # DEBUG: check correctness
            # (output_lb2, output_ub2), output_coeffs2 = verifier.abstractor.compute_bounds(
            #     input_lowers=self.input_output_bounds[idx].over_input[0],
            #     input_uppers=self.input_output_bounds[idx].over_input[1],
            #     cs=objective.cs if idx==n_subnets-1 else None,
            #     method=method,
            # )
            # assert torch.allclose(output_ub, output_ub2), f'{torch.norm(output_ub - output_ub2)}'
            # print('Pass ub', torch.norm(output_ub - output_ub2))
            # assert torch.allclose(output_lb, output_lb2), f'{torch.norm(output_lb - output_lb2)}'
            # print('Pass lb', torch.norm(output_lb - output_lb2))
            
            # for field in ['lA', 'uA', 'lbias', 'ubias']:
            #     f1 = getattr(output_coeffs, field)
            #     f2 = getattr(output_coeffs2, field)
            #     assert torch.allclose(f1, f2, atol=1e-5), f'{torch.norm(f1 - f2)}'
            #     print(f'Pass {field}', torch.norm(f1 - f2))
                
            
            if idx == 0 and use_extra: 
                # TODO: generalize for more than 2 subnets
                # additional backsub up to the original input 
                self.extras[idx + 1] = {
                    'input': (input_lb_0.clone(), input_ub_0.clone()),
                    'coeff': output_coeffs
                }
            
            # flatten output
            subnet_params = self.sub_networks[idx]
            if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
                assert len(output_lb) == len(output_ub) == 1
                output_lb = output_lb.view(subnet_params.output_shape)
                output_ub = output_ub.view(subnet_params.output_shape)

            # print(f'{output_lb=}')
            print(f'{output_lb.shape=}')
            
            # update bounds
            assert self.input_output_bounds[idx].over_output is None
            self.input_output_bounds[idx] = self.input_output_bounds[idx]._replace(over_output=(output_lb.clone(), output_ub.clone()))
            assert self.input_output_bounds[idx].over_output is not None
            
            if self.input_output_bounds.get(idx + 1, False) is None:
                self.input_output_bounds[idx + 1] = InOutBounds(over_input=(output_lb.clone(), output_ub.clone()))
        
        
        return self.input_output_bounds[n_subnets-1].over_output
    
    @beartype
    def _verify_subnet(self, subnet_idx: int, objective: typing.Any, verify_batch: int, timeout: int | float = 20.0) -> str:
        # release memory
        gc_cuda()
            
        subnet_input_outputs = self.input_output_bounds[subnet_idx]
        class TMP:
            pass
        
        assert torch.all(subnet_input_outputs.over_input[0] < subnet_input_outputs.over_input[1])
        tmp_objective = TMP()
        tmp_objective.lower_bounds = subnet_input_outputs.over_input[0].clone()
        tmp_objective.upper_bounds = subnet_input_outputs.over_input[1].clone()
        tmp_objective.lower_bounds_f64 = tmp_objective.lower_bounds.to(torch.float64)
        tmp_objective.upper_bounds_f64 = tmp_objective.upper_bounds.to(torch.float64)
        tmp_objective.ids = objective.ids
        tmp_objective.cs = objective.cs
        tmp_objective.rhs = objective.rhs
        tmp_objective.cs_f64 = objective.cs.to(torch.float64)
        tmp_objective.rhs_f64 = objective.rhs.to(torch.float64)

        # tmp_objective2 = TMP()
        # tmp_objective2.lower_bounds = subnet_input_outputs.over_input[0].clone()
        # tmp_objective2.upper_bounds = subnet_input_outputs.over_input[1].clone()
        # tmp_objective2.lower_bounds_f64 = tmp_objective2.lower_bounds.to(torch.float64)
        # tmp_objective2.upper_bounds_f64 = tmp_objective2.upper_bounds.to(torch.float64)
        # tmp_objective2.ids = objective.ids
        # tmp_objective2.cs = objective.cs
        # tmp_objective2.rhs = objective.rhs
        # tmp_objective2.cs_f64 = objective.cs.to(torch.float64)
        # tmp_objective2.rhs_f64 = objective.rhs.to(torch.float64)
        
        assert torch.all(tmp_objective.lower_bounds < tmp_objective.upper_bounds)

        verifier = self._setup_subnet_verifier(
            subnet_idx=subnet_idx, 
            objective=tmp_objective, 
            batch=verify_batch,
        )
        print(verifier.net)

        verifier.start_time = time.time()

        # verifier.abstractor.extras = None
        status = verifier._verify_one(
            objective=tmp_objective, 
            preconditions={}, 
            reference_bounds={}, 
            timeout=timeout,
        )
        
        print(f'{status=}')

        return status
    
    def attack_subnet(self, model, objective, timeout=5.0):
        #     tmp_objective = copy.deepcopy(objective)
        #     # FIXME: generalize to more than 2 subnetworks
        #     assert subnet_idx == 1, f'Invalid {subnet_idx=}'
        #     # update the input bounds
        #     tmp_objective.lower_bounds = self.input_output_bounds[subnet_idx].over_input[0].clone()
        #     tmp_objective.upper_bounds = self.input_output_bounds[subnet_idx].over_input[1].clone()
        print(model)
        print(f'{objective.lower_bounds.shape=}')
        input_lower = objective.lower_bounds.to(self.device)
        input_upper = objective.upper_bounds.to(self.device)
        assert torch.all(input_lower < input_upper)
        
        for i in range(1):
            x_attack = (input_upper - input_lower) * torch.rand(input_lower.shape, device=self.device) + input_lower
            print(f'Trial {i}: {x_attack.sum().item()=}')
            pred = model(x_attack).cpu().detach()
            
            write_vnnlib(
                spec_path='example/vnnlib/spec_dec.vnnlib',
                data_lb=input_lower,
                data_ub=input_upper,
                prediction=pred
            )
            
            torch.onnx.export(
                model,
                x_attack,
                'example/onnx/net_dec.onnx',
                verbose=False,
                opset_version=12,
            )
            
            # exit()
            cs = objective.cs.to(self.device)
            rhs = objective.rhs.to(self.device)
            data_min_attack = input_lower.unsqueeze(1).expand(-1, len(cs), *input_lower.shape[1:])
            data_max_attack = input_upper.unsqueeze(1).expand(-1, len(cs), *input_upper.shape[1:])
            assert torch.all(data_min_attack < data_max_attack)
            # print(rhs.shape, cs.shape, data_max_attack.shape, input_upper.shape)
            is_attacked, attack_images = pgd_attack(
                model=model,
                x=x_attack, 
                data_min=data_min_attack,
                data_max=data_max_attack,
                cs=cs,
                rhs=rhs,
                attack_iters=500, 
                num_restarts=20,
                timeout=timeout,
                use_gama=False,
            )
            print(f'Trial {i}: {is_attacked=}\n')
            if is_attacked:
                raise
        exit()
        
    @beartype
    def _under_estimate(self, subnet_idx: int, verify_batch: int, verify_timeout: int | float, tighten_batch: int) -> None:
        # release memory
        gc_cuda()
            
        subnet_input_outputs = self.input_output_bounds[subnet_idx]
        subnet_params = self.sub_networks[subnet_idx]
        # network
        if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
            network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
        else:
            # network = torch.nn.Sequential(torch.nn.Identity(), subnet_params.network)
            network = subnet_params.network
        
        if subnet_input_outputs.under_output is None:
            min_i = optimize_dnn(network, subnet_input_outputs.over_input[0], subnet_input_outputs.over_input[1], is_min=True).view(subnet_params.output_shape)
            max_i = optimize_dnn(network, subnet_input_outputs.over_input[0], subnet_input_outputs.over_input[1], is_min=False).view(subnet_params.output_shape)
            assert torch.all(min_i <= max_i), f'{min_i=} {max_i=}'
            self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(under_output=(min_i.clone(), max_i.clone()))
            subnet_input_outputs = self.input_output_bounds[subnet_idx]
            if os.environ.get("NEURALSAT_LOG_SUBVERIFIER"):
                print(f'Setup under output {subnet_idx=}:')
                print(f'\t- Lower: {subnet_input_outputs.under_output[0].detach().cpu().numpy().tolist()}')
                print(f'\t- Upper: {subnet_input_outputs.under_output[1].detach().cpu().numpy().tolist()}')

        candidate_neurons = self.extract_candidate(
            subnet_idx=subnet_idx, 
            batch=tighten_batch,
        )
        
        verifier = self._setup_subnet_verifier(
            subnet_idx=subnet_idx,
            objective=None, 
            batch=verify_batch,
        )
        
        verified_candidates, attack_samples = verify_dnf_pairs(
            verifier=verifier,
            input_lower=subnet_input_outputs.over_input[0],
            input_upper=subnet_input_outputs.over_input[1],
            n_outputs=subnet_input_outputs.under_output[0].shape[-1],
            candidate_neurons=candidate_neurons,
            batch=10,
            timeout=verify_timeout,
        )
        # print(f'{len(verified_candidates)=}')
        # FIXME: handle attack
        # if len(attack_samples) and subnet_idx==0:
        #     print(f'Attacked {len(attack_samples)=} {subnet_input_outputs.over_input[0].sum().item()=} {subnet_input_outputs.over_input[1].sum().item()=}')
        #     assert torch.all(attack_samples <= subnet_input_outputs.over_input[1])
        #     assert torch.all(attack_samples >= subnet_input_outputs.over_input[0])
        #     attack_outputs = network(attack_samples)
        #     # print(f'{attack_outputs=}')
        #     attack_outputs_min = attack_outputs.amin(dim=0)
        #     attack_outputs_max = attack_outputs.amax(dim=0)
        #     # print(f'{attack_outputs_min=}')
        #     # print(f'{attack_outputs_max=}')
        #     old_under_output_lower, old_under_output_upper = subnet_input_outputs.under_output
        #     # print(f'{old_under_output_lower=}')
        #     # print(f'{old_under_output_upper=}')
        #     indices_min = attack_outputs_min < old_under_output_lower
        #     indices_max = attack_outputs_max > old_under_output_upper
        #     print(f'{indices_min=}')
        #     print(f'Update under_output lower {subnet_idx=}:')
        #     for idx, iv in enumerate(indices_min.flatten()):
        #         if iv:
        #             print(f'[{idx}] {old_under_output_lower.flatten()[idx]} => {attack_outputs_min.flatten()[idx]}')
        #     print()
        #     print(f'Update under_output upper {subnet_idx=}:')
        #     for idx, iv in enumerate(indices_max.flatten()):
        #         if iv:
        #             print(f'[{idx}] {old_under_output_upper.flatten()[idx]} => {attack_outputs_max.flatten()[idx]}')
        #     # exit()
        #     new_under_output_lower = torch.where(indices_min, attack_outputs_min, old_under_output_lower)
        #     new_under_output_upper = torch.where(indices_max, attack_outputs_max, old_under_output_upper)
            
        #     assert torch.all(new_under_output_lower <= new_under_output_upper)
        #     self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(under_output=(new_under_output_lower.clone(), new_under_output_upper.clone()))
        #     subnet_input_outputs = self.input_output_bounds[subnet_idx]
        
        new_over_output_lower = subnet_input_outputs.over_output[0].clone()
        new_over_output_upper = subnet_input_outputs.over_output[1].clone()
        
        # print(f'{new_over_output_lower.shape=}')
        improved_neuron_indices = []
        # improved_lower, improved_upper = 0.0, 0.0
        for (neuron_idx, neuron_bound, neuron_direction) in verified_candidates:
            assert neuron_direction in ['lt', 'gt']
            if neuron_direction == 'lt': # lower bound
                # assert new_over_output_lower[0][neuron_idx] <= neuron_bound, f'{neuron_idx=} {new_over_output_lower[0][neuron_idx]=} <= {neuron_bound=}'
                # improved_lower += abs(new_over_output_lower[0][neuron_idx] - neuron_bound)
                new_over_output_lower[0][neuron_idx] = max(new_over_output_lower[0][neuron_idx], neuron_bound)
            else: # upper bound
                # assert new_over_output_upper[0][neuron_idx] >= neuron_bound, f'{neuron_idx=} {new_over_output_upper[0][neuron_idx]=} >= {neuron_bound=}'
                # improved_upper += abs(new_over_output_upper[0][neuron_idx] - neuron_bound)
                new_over_output_upper[0][neuron_idx] = min(new_over_output_upper[0][neuron_idx], neuron_bound)
            improved_neuron_indices.append(neuron_idx)
        
        for neuron_idx in list(sorted(set(improved_neuron_indices))):
            print(
                f'Tightened {neuron_idx=:3d}:\t'
                f'[{subnet_input_outputs.over_output[0][0][neuron_idx]:.04f}, {subnet_input_outputs.over_output[1][0][neuron_idx]:.04f}]\t'
                f'=>\t[{new_over_output_lower[0][neuron_idx]:.04f}, {new_over_output_upper[0][neuron_idx]:.04f}]'
            )
            
        # update bounds
        self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(over_output=(new_over_output_lower.clone(), new_over_output_upper.clone()))
        self.input_output_bounds[subnet_idx+1] = self.input_output_bounds[subnet_idx+1]._replace(over_input=(new_over_output_lower.clone(), new_over_output_upper.clone()))
            
        
    @beartype
    def extract_candidate(self, subnet_idx: int, batch: int, eps: float=0.0) -> list:
        if (subnet_idx not in self.tightening_candidates) or len(self.tightening_candidates[subnet_idx]) < batch // 4:
            subnet_input_outputs = self.input_output_bounds[subnet_idx]
            under_output = subnet_input_outputs.under_output
            over_output = subnet_input_outputs.over_output
            
            assert over_output is not None, f'Unsupported {over_output=}'
            assert under_output is not None, f'Unsupported {under_output=}'
            
            best_interm_min = under_output[0].flatten()
            best_interm_max = under_output[1].flatten()
            
            over_output_min = over_output[0].flatten()
            over_output_max = over_output[1].flatten()
            assert torch.all(best_interm_min <= best_interm_max)
            assert torch.all(over_output_min <= over_output_max)
            
            assert torch.all(over_output_min <= best_interm_min), f'{over_output_min=} {best_interm_min=} {over_output_min < best_interm_min}'
            assert torch.all(over_output_max >= best_interm_max), f'{over_output_max=} {best_interm_max=} {over_output_max < best_interm_max}'

            candidates = []
            for i in range(len(best_interm_min)):
                if (over_output_min[i] * over_output_max[i] < 0) or 1: # unstable neurons
                    candidates.append([(i, (best_interm_min[i] + over_output_min[i]) / 2 + eps, 'lt')])
                    candidates.append([(i, (best_interm_max[i] + over_output_max[i]) / 2 - eps, 'gt')])
                    print(f'[{over_output_min[i]:.04f}, {over_output_max[i]:.04f}],\t[{best_interm_min[i]:.04f}, {best_interm_max[i]:.04f}]\t=>\t{candidates[-2:]}')
                    
            random.shuffle(candidates)
            self.tightening_candidates[subnet_idx] = candidates
        
        assert len(self.tightening_candidates[subnet_idx]) > 0
        candidates = self.tightening_candidates[subnet_idx][:batch]
        print(f'Extracted {len(candidates)=} from {len(self.tightening_candidates[subnet_idx])} candidates')
        self.tightening_candidates[subnet_idx] = self.tightening_candidates[subnet_idx][len(candidates):]
    
        return candidates
        
    @beartype
    def verify_one(self, objective: typing.Any, verify_batch: int, tighten_batch: int, timeout: int | float = 3600, use_extra: bool = True) -> str:
        self.start_time = time.time()
        self.reset()
        
        lb, _ = self._init_interm_bounds(objective, use_extra=use_extra, method='crown-optimized')
        print(f'{lb=}')
        exit()
        if lb >= 0.0:
            return ReturnStatus.UNSAT
        
        # release memory
        gc_cuda()
        
        self.iteration = 0
        n_subnets = len(self.sub_networks)
        while True:
            print('[+] Iteration:', self.iteration)
            for subnet_idx in range(n_subnets-1):
                self._under_estimate(
                    subnet_idx=subnet_idx,
                    verify_batch=verify_batch,
                    verify_timeout=10.0,
                    tighten_batch=tighten_batch,
                )
                if os.environ.get("NEURALSAT_LOG_SUBVERIFIER"):
                    print('###################')
                    print(f'Bounds {subnet_idx=}')
                    subnet_input_outputs = self.input_output_bounds[subnet_idx]
                    print(f'Over output: {subnet_input_outputs.over_output[0].shape}')
                    print(f'\t- Lower: {subnet_input_outputs.over_output[0].detach().cpu().numpy().tolist()}')
                    print(f'\t- Upper: {subnet_input_outputs.over_output[1].detach().cpu().numpy().tolist()}')
                    print(f'Under output:')
                    print(f'\t- Lower: {subnet_input_outputs.under_output[0].detach().cpu().numpy().tolist()}')
                    print(f'\t- Upper: {subnet_input_outputs.under_output[1].detach().cpu().numpy().tolist()}')
                
                
            # last subnet
            status = self._verify_subnet(
                subnet_idx=n_subnets-1, 
                objective=copy.deepcopy(objective), 
                verify_batch=verify_batch,
                timeout=20.0,
            )
            # exit()
            
            if status not in [ReturnStatus.TIMEOUT, ReturnStatus.SAT]:
                return status
            
            self.iteration += 1
            if time.time() - self.start_time > timeout:
                return status
    
    @beartype
    def extract_tokenizer(self, net: nn.Module, objectives: DnfObjectives, input_shape: tuple) -> tuple[nn.Module, typing.Any, tuple]:
        return net, objectives, input_shape
        if not hasattr(net, 'tokenizer'):
            return net, objectives, input_shape
        
        # extract tokenizer
        # step 1: verify correctness
        dummy = torch.randn(2, *input_shape[1:], device=self.device)
        y1 = net(dummy)
        
        assert len(net.classifier) == 2 # TODO: allow more than 2 subnets
        new_net = PytorchWrapper(net.classifier)
        y2 = new_net(net.tokenizer(dummy))
        
        if not torch.allclose(y1, y2):
            return net, objectives, input_shape
            
        print('[Extract Tokenizer] Correctness checking: Pass')
        
        # step 2: initialize abstract domain
        tokenizer_abstractor = BoundedModule(
            model=net.tokenizer, 
            global_input=torch.zeros(input_shape, device=self.device),
            bound_opts={'conv_mode': 'patches', 'verbosity': 0},
            device=self.device,
            verbose=False,
        )
        tokenizer_abstractor.eval()
        
        # step 3: initialize intervals
        input_lowers = objectives.lower_bounds.view(-1, *input_shape[1:])
        input_uppers = objectives.upper_bounds.view(-1, *input_shape[1:])
        assert torch.all(input_lowers < input_uppers)
        x = self.new_input(x_L=input_lowers, x_U=input_uppers)
        assert not hasattr(x.ptb, 'extras')
        
        # step 3: forward intervals
        token_lowers, token_uppers = tokenizer_abstractor.compute_bounds(
            x=(x,), 
            method='backward', 
            bound_upper=True,
        )
        assert torch.all(token_lowers < token_uppers)
        # print(tokenizer_abstractor)
        # print(f'{input_lowers.shape=}')
        # print(f'{token_lowers.shape=}')
        
        # step 5: convert objectives
        objectives.lower_bounds = token_lowers.flatten(1).to(objectives.lower_bounds)
        objectives.upper_bounds = token_uppers.flatten(1).to(objectives.upper_bounds)
        assert torch.all(objectives.lower_bounds < objectives.upper_bounds)
        
        objectives.lower_bounds_f64 = token_lowers.flatten(1).to(objectives.lower_bounds_f64)
        objectives.upper_bounds_f64 = token_uppers.flatten(1).to(objectives.upper_bounds_f64)
        assert torch.all(objectives.lower_bounds_f64 < objectives.upper_bounds_f64)
        
        new_input_shape = (1,) + token_lowers.shape[1:]
        return new_net, objectives, new_input_shape
    
    @beartype
    def decompositional_verify(self, objectives: DnfObjectives, timeout: int |float = 3600, batch: int = 500) -> str:
        # decomposition
        # step 1: Extract Tokenizer ViT (if needed) + Convert objectives
        new_network, new_objectives, new_input_shape = self.extract_tokenizer(
            net=self.net, 
            objectives=copy.deepcopy(objectives), 
            input_shape=self.input_shape,
        )
        gc_cuda()
        
        # step 2: Extract Prefix + Suffix of Classifier     
        self.decompose_network(
            net=new_network, 
            input_shape=new_input_shape,
            min_layer=self.min_layer,
        )
        self.reset()
        
        # step 3: Verify each objecive
        while len(new_objectives):
            objective = new_objectives.pop(1)
            status = self.verify_one(
                objective=objective,
                verify_batch=batch, # batch size of sub-verifiers
                tighten_batch=50, # number of tightening candidates
                timeout=timeout,
                use_extra=False,
            )
            break # TODO: remove
            
            if status in [ReturnStatus.SAT, ReturnStatus.TIMEOUT, ReturnStatus.UNKNOWN]:
                return status 
            if status == ReturnStatus.UNSAT:
                continue
            raise ValueError(status)
        
            
        return ReturnStatus.UNSAT  
    
    
    @beartype
    def original_verify(self, objectives: DnfObjectives, timeout: int | float = 3600, batch: int = 500) -> str:
        
        verifier = Verifier(
            net=self.net,
            input_shape=self.input_shape,
            batch=batch,
            device=self.device,
        )
        status = verifier.verify(
            dnf_objectives=copy.deepcopy(objectives), 
            timeout=timeout, 
            force_split=None,
        )
        
        return status
        
    @beartype
    def verify(self, objectives: DnfObjectives, force_decomposition: bool = False, timeout: int | float = 3600, batch: int = 500) -> str:
        if force_decomposition:
            return self.decompositional_verify(
                objectives=objectives,
                timeout=timeout,
                batch=batch,
            )
            
        try:
            # Try verify entire network
            return self.original_verify(
                objectives=objectives,
                timeout=timeout,
                batch=batch,
            )
        except RuntimeError as exception:
            if is_cuda_out_of_memory(exception):
                print('[Debug] Switch to decompositional verification due to OOM')
                return self.decompositional_verify(objectives)
            else:
                traceback.print_exc()
                raise NotImplementedError()
        except SystemExit:
            exit()
        except:
            traceback.print_exc()
            raise NotImplementedError()


    @beartype
    def new_input(self, x_L: torch.Tensor, x_U: torch.Tensor) -> BoundedTensor:
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(x_L <= x_U + 1e-8) #, f'{x_L=}\n\n{x_U=}'
        new_x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(self.device)
        if hasattr(self, 'extras'):
            new_x.ptb.extras = self.extras
        return new_x        

def formatted_print(a, b, name):
    if a.numel() > 5:
        a = ', '.join([f'{_.item():.03f}' for _ in a.flatten()])
        b = ', '.join([f'{_.item():.03f}' for _ in b.flatten()])
    print(f'[{name}] lb:', a)
    print(f'[{name}] ub:', b)
    print()
    

def test1():    
    
    seed = int(sys.argv[1])
    print(f'{seed=}')
    torch.manual_seed(seed)
    
    from example.scripts.test_function import extract_instance
    logger.setLevel(logging.DEBUG)
    
    Settings.setup(None)
    print(Settings)
    
    # net_path = 'example/onnx/mnist-net_142x9.onnx'
    # vnnlib_path = 'example/vnnlib/prop_8_0.03.vnnlib'
    
    # sat
    net_path = 'example/onnx/mnist-net_256x4.onnx'
    vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
    
    # net_path = 'example/onnx/cifar10_2_255_simplified.onnx'
    # vnnlib_path = 'example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib'
    
    # net_path = 'example/onnx/net_conv_small.onnx'
    # vnnlib_path = 'example/vnnlib/net_conv_small.vnnlib'
    
    net_path = 'example/onnx/mnist-net_256x6.onnx'
    vnnlib_path = 'example/vnnlib/prop_8_0.03.vnnlib'
    
    # net_path = 'example/onnx/net_relu_random.onnx'
    # vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
    
    # net_path = 'example/onnx/mnist-fnn-vae.onnx'
    # vnnlib_path = 'example/vnnlib/prop_8_0.03.vnnlib'
    model_name = 'resnet20B'
    benchmark_dir = 'example/generated_benchmark/resnet_b'
    eps = 0.01
    
    instances = [l.split(',')[:-1] for l in open(f'{benchmark_dir}/eps_{eps:.06f}_{model_name}/instances.csv').read().strip().split('\n')]
    # print(instances)
    
    net_path = f'{benchmark_dir}/eps_{eps:.06f}_{model_name}/{instances[seed][0]}'
    vnnlib_path = f'{benchmark_dir}/eps_{eps:.06f}_{model_name}/{instances[seed][1]}'
    
    
    device = 'cuda'
    # device = 'cpu'
    # split_idx = 3
    # use_extra = True
    
    print(f'{net_path=}')
    pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
    # print(pytorch_model)
    # exit()
    pytorch_model = resnet20B()
    pytorch_model.eval()
    print(pytorch_model)
    # exit()
    # torch.onnx.export(
    #     pytorch_model,
    #     torch.ones(input_shape),
    #     'example/onnx/net_dec.onnx',
    #     verbose=False,
    #     opset_version=12,
    # )
    # print('Export: example/onnx/net_dec.onnx')
    # print(pytorch_model)
    # print(f'{input_shape=}')
    
    if 0:
        verifier = DecompositionalVerifier(
            net=pytorch_model,
            input_shape=input_shape,
            min_layer=40,
            device=device,
        )    
        
        
        tic = time.time()
        objective = dnf_objectives.pop(1)
        lb, ub = verifier._init_interm_bounds(objective)
        print(time.time() - tic)
        assert torch.all(lb <= ub)
        formatted_print(lb, ub, 'Full')
        
        
        # verifier = DecompositionalVerifier(
        #     net=pytorch_model,
        #     input_shape=input_shape,
        #     min_layer=split_idx,
        #     device=device,
        # )    
        
        # tic = time.time()
        # lb, ub = verifier._init_interm_bounds(objective, use_extra=False)
        # print(time.time() - tic)
        # assert torch.all(lb <= ub)
        # formatted_print(lb, ub, 'Split (False)')
        
      
    if 0:  
        layers = nn.ModuleList([
            nn.Sequential(nn.Flatten(1), nn.Linear(784, 13), nn.ReLU(), nn.Linear(13, 14)),
            nn.Sequential(nn.ReLU(), nn.Linear(14, 15)),
            nn.Sequential(nn.ReLU(), nn.Linear(15, 10)),
        ])
            
        pytorch_model = PytorchWrapper(layers)
        print(pytorch_model)
    
    
    verifier = DecompositionalVerifier(
        net=pytorch_model,
        input_shape=input_shape,
        min_layer=3,
        device=device,
    )    

    tic = time.time()
    # lb, ub = verifier._init_interm_bounds(objective, use_extra=use_extra)
    # print(time.time() - tic)
    # assert torch.all(lb <= ub)
    # formatted_print(lb, ub, f'Split ({use_extra=})')
    
    # status = verifier.verify_one(
    #     objective=objective,
    #     verify_batch=500, # batch size of sub-verifiers
    #     tighten_batch=50, # number of tightening candidates
    #     timeout=3600,
    #     use_extra=True,
    # )
    
    oracle_verify = verifier.decompositional_verify
    # oracle_verify = verifier.original_verify
    
    status = oracle_verify(
        objectives=dnf_objectives, 
        timeout=10000,
        batch=500,
    )
    
    print(status, time.time() - tic)
    
    
# def test2():
    
#     from example.scripts.test_function import extract_instance
#     from train.models.vit.vit import vit_6_8_384
    
    
#     net_path = 'example/onnx/cifar10_2_255_simplified.onnx'
#     vnnlib_path = 'example/vnnlib/cifar10_spec_idx_4_eps_0.00784_n1.vnnlib'
#     _, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
#     pytorch_model = vit_6_8_384()
    
#     split_idx = 2
#     device = 'cpu'
#     device = 'cuda'
        
#     verifier = DecompositionalVerifier(
#         net=pytorch_model,
#         input_shape=input_shape,
#         min_layer=split_idx,
#         device=device,
#     )    
    
#     new_net, new_objs, new_shapes = verifier.extract_tokenizer(pytorch_model, dnf_objectives, input_shape)
#     print(new_net)
    
#     while len(new_objs):
#         obj = new_objs.pop(1)
#         output = new_net(obj.lower_bounds.view(-1, *new_shapes[1:]).to(device))
#         print(f'{new_shapes=} {obj.lower_bounds.shape=} {obj.cs.shape=} {output.shape=}')
        
        
#     verifier.decompose_network(
#         net=new_net,
#         input_shape=new_shapes,
#         min_layer=1,
#     )
    
#     print(verifier.sub_networks)
    
if __name__ == "__main__":
    test1()