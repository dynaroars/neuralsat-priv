from collections import namedtuple
import logging
import random
import torch
import time
import copy
import os

from util.network.read_onnx import parse_onnx, decompose_pytorch
from util.spec.write_vnnlib import write_vnnlib

from util.misc.torch_cuda_memory import gc_cuda
from util.misc.result import ReturnStatus
from util.misc.logger import logger

from attacker.pgd_attack.general import attack as pgd_attack
from tightener.utils import optimize_dnn, verify_dnf_pairs
from verifier.verifier import Verifier

from setting import Settings



SubNetworks = namedtuple('SubNetworks', ['network', 'input_shape', 'output_shape'])
InOutBounds = namedtuple(
    'InputOutputBounds', [
        'under_input', 
        'under_output',
        'over_input', 
        'over_output',
    ],
    defaults=(None,) * 4
)

class DecompositionalVerifier:
    
    def __init__(self, net, input_shape: tuple, min_layer: int, device: str = 'cpu') -> None:
        self.net = net.to(device) # pytorch model
        self.device = device
        self.input_shape = input_shape
        
        self._setup_sub_networks(min_layer=min_layer)
        self.reset()
        
    def reset(self):
        self.input_output_bounds = {k: None for k in range(len(self.sub_networks))}
        self.extras = {0: None}
        self.tightening_candidates = {}
        
    def _setup_sub_networks(self, min_layer):
        self.sub_networks = {}
        
        input_net = self.net
        input_shape = self.input_shape
        count = 0
        while True:
            prefix, suffix = self._split_network(input_net, input_shape, min_layer=min_layer)
            self.sub_networks[count] = SubNetworks(*prefix)
            if suffix is None:
                break 
            input_net, input_shape, _ = suffix
            count += 1
            # FIXME: support more than 2
            min_layer = 10000
        
        # FIXME: support more than 2
        assert len(self.sub_networks) <= 2
        
        # check correctness
        dummy = torch.randn(2, *self.input_shape[1:], device=self.device)
        output_1 = self.net(dummy)
        output_2 = dummy.clone()
        for idx in range(len(self.sub_networks)):
            output_2 = self.sub_networks[idx].network(output_2)
        assert torch.allclose(output_1, output_2)
        print(f'[+] Passed {len(self.sub_networks)=}')

    def _split_network(self, net, input_shape, min_layer: int):
        split_idx = min_layer
        while True:
            prefix_onnx_byte, suffix_onnx_byte = decompose_pytorch(net, input_shape, split_idx + 1)
            net = net.to(self.device)
            if (prefix_onnx_byte is None) or (suffix_onnx_byte is None):
                return (net, input_shape, None), None
            assert prefix_onnx_byte is not None

            # move to next layer
            split_idx += 1
            
            # parse subnets
            prefix, prefix_input_shape, prefix_output_shape, _ = parse_onnx(prefix_onnx_byte)
            suffix, suffix_input_shape, suffix_output_shape, _ = parse_onnx(suffix_onnx_byte)
            # TODO: check if prefix_output_shape is too large
            
            # if len(list(suffix.children())) <= min_layer:
            #     return (net, input_shape, None), None
            
            # move to device
            prefix = prefix.to(self.device)
            suffix = suffix.to(self.device)
                
            # check correctness
            dummy = torch.randn(2, *prefix_input_shape[1:], device=self.device) # try batch=2
            if torch.allclose(net(dummy), suffix(prefix(dummy))):
                return (prefix.to(self.device), prefix_input_shape, prefix_output_shape), (suffix.to(self.device), suffix_input_shape, suffix_output_shape)
            
    def _setup_subnet_verifier(self, subnet_idx, objective=None, batch=500):
        subnet_params = self.sub_networks[subnet_idx]
        # network
        if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
            network = torch.nn.Sequential(subnet_params.network, torch.nn.Flatten(1))
        else:
            network = torch.nn.Sequential(torch.nn.Identity(), subnet_params.network)

        verifier = Verifier(
            net=network,
            input_shape=subnet_params.input_shape,
            batch=batch,
            device=self.device,
        )
        verifier._setup_restart_naive(0, objective)
        verifier.abstractor.extras = self.extras.get(subnet_idx, None)
        return verifier

    def _init_interm_bounds(self, objective, use_extra=True):
        print('Init interm bounds')
        # input 
        input_lb_0 = objective.lower_bounds.view(self.input_shape).to(self.device)
        input_ub_0 = objective.upper_bounds.view(self.input_shape).to(self.device)
        self.input_output_bounds[0] = InOutBounds(over_input=(input_lb_0.clone(), input_ub_0.clone()))
        n_subnets = len(self.sub_networks)
        for idx in range(n_subnets):
            print(f'Processing subnet {idx=}/{n_subnets}')
            verifier = self._setup_subnet_verifier(idx)
            # print(verifier.net)
            
            (output_lb, output_ub), output_coeffs = verifier.abstractor.compute_bounds(
                input_lowers=self.input_output_bounds[idx].over_input[0],
                input_uppers=self.input_output_bounds[idx].over_input[1],
                cs=objective.cs if idx==n_subnets-1 else None,
                # method='backward',
                method='crown-optimized'
            )
            
            if idx == 0 and use_extra: 
                self.extras[idx + 1] = {
                    'input': (input_lb_0.clone(), input_ub_0.clone()),
                    'coeff': output_coeffs
                }
            
            # flatten output
            subnet_params = self.sub_networks[idx]
            if (subnet_params.output_shape is not None) and len(subnet_params.output_shape) > 2:
                output_lb = output_lb.view(subnet_params.output_shape)
                output_ub = output_ub.view(subnet_params.output_shape)

            print(f'{output_lb.shape=}')
            # update bounds
            assert self.input_output_bounds[idx].over_output is None
            self.input_output_bounds[idx] = self.input_output_bounds[idx]._replace(over_output=(output_lb.clone(), output_ub.clone()))
            assert self.input_output_bounds[idx].over_output is not None
            
            if self.input_output_bounds.get(idx + 1, False) is None:
                self.input_output_bounds[idx + 1] = InOutBounds(over_input=(output_lb.clone(), output_ub.clone()))

        # release memory
        gc_cuda()
            
        return self.input_output_bounds[n_subnets-1].over_output
    
    def _verify_subnet(self, subnet_idx, objective, verify_batch, timeout=20.0):
        # release memory
        gc_cuda()
            
        subnet_input_outputs = self.input_output_bounds[subnet_idx]
        class TMP:
            pass
        
        assert torch.all(subnet_input_outputs.over_input[0] < subnet_input_outputs.over_input[1])
        tmp_objective = TMP()
        tmp_objective.lower_bounds = subnet_input_outputs.over_input[0].clone()
        tmp_objective.upper_bounds = subnet_input_outputs.over_input[1].clone()
        tmp_objective.ids = objective.ids
        tmp_objective.cs = objective.cs
        tmp_objective.rhs = objective.rhs
        # print(f'{subnet_input_outputs.over_input[0].shape=} {tmp_objective.upper_bounds.shape=}')
        
        # mask = torch.where(tmp_objective.lower_bounds * tmp_objective.upper_bounds < 0)
        # print(f'{mask=}')
        # print(f'{tmp_objective.lower_bounds.shape=} {tmp_objective.lower_bounds.sum()=} {tmp_objective.upper_bounds.sum()=}')
        # mask_idx = 12
        # FIXME: setting lower_bound=0 makes all the bounds worse!!!
        # print(tmp_objective.lower_bounds[mask[0][mask_idx], mask[1][mask_idx]], tmp_objective.upper_bounds[mask[0][mask_idx], mask[1][mask_idx]])
        # tmp_objective.lower_bounds[mask[0][mask_idx], mask[1][mask_idx]] = 0.
        # tmp_objective.upper_bounds[mask[0][mask_idx], mask[1][mask_idx]] = 0.
        # tmp_objective.upper_bounds[mask] = 0.
        # tmp_objective.lower_bounds[mask] = 0.
        assert torch.all(tmp_objective.lower_bounds < tmp_objective.upper_bounds)
        # print('passed')
        # print(f'{tmp_objective.lower_bounds.shape=} {tmp_objective.lower_bounds.sum()=} {tmp_objective.upper_bounds.sum()=}')
        # print(f'{tmp_objective.lower_bounds=}')
        # print(f'{tmp_objective.upper_bounds=}')
        verifier = self._setup_subnet_verifier(
            subnet_idx=subnet_idx, 
            objective=copy.deepcopy(tmp_objective), 
            batch=verify_batch,
        )
        print(verifier.net)
        # self.attack_subnet(verifier.net, copy.deepcopy(tmp_objective))
        # exit()
        verifier.start_time = time.time()
        # print(tmp_objective)
        # verifier.abstractor.extras = None
        status = verifier._verify_one(
            objective=copy.deepcopy(tmp_objective), 
            preconditions={}, 
            reference_bounds={}, 
            timeout=timeout,
        )
        
        print(f'{status=}')
        # (output_lb, output_ub), output_coeffs = verifier.abstractor.compute_bounds(
        #     input_lowers=self.input_output_bounds[subnet_idx].over_input[0],
        #     input_uppers=self.input_output_bounds[subnet_idx].over_input[1],
        #     cs=objective.cs,
        #     method='backward',
        #     # method='crown-optimized'
        # )
        # print(f'{output_lb=}')
        # print(f'{output_ub=}')
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
        
    def _under_estimate(self, subnet_idx, verify_batch, verify_timeout, tighten_batch):
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
            assert torch.all(min_i < max_i)
            self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(under_output=(min_i.clone(), max_i.clone()))
            subnet_input_outputs = self.input_output_bounds[subnet_idx]

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
        assert not len(attack_samples), f'Attacked'
        
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
            print(f'Tightened {neuron_idx=:3d}:\t'
                    f'[{subnet_input_outputs.over_output[0][0][neuron_idx]:.04f}, {subnet_input_outputs.over_output[1][0][neuron_idx]:.04f}]\t'
                    f'=>\t[{new_over_output_lower[0][neuron_idx]:.04f}, {new_over_output_upper[0][neuron_idx]:.04f}]'
            )
            
        # update bounds
        self.input_output_bounds[subnet_idx] = subnet_input_outputs._replace(over_output=(new_over_output_lower.clone(), new_over_output_upper.clone()))
        self.input_output_bounds[subnet_idx+1] = self.input_output_bounds[subnet_idx+1]._replace(over_input=(new_over_output_lower.clone(), new_over_output_upper.clone()))
            
        
    def extract_candidate(self, subnet_idx, batch, eps=0.0):
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
            assert torch.all(best_interm_min < best_interm_max)
            assert torch.all(over_output_min < over_output_max)
            
            assert torch.all(over_output_min < best_interm_min)
            assert torch.all(over_output_max > best_interm_max)

            candidates = []
            for i in range(len(best_interm_min)):
                if (over_output_min[i] * over_output_max[i] < 0) or 1: # unstable neurons
                    candidates.append([(i, (best_interm_min[i] + over_output_min[i]) / 2 + eps, 'lt')])
                    candidates.append([(i, (best_interm_max[i] + over_output_max[i]) / 2 - eps, 'gt')])
                    print(f'[{over_output_min[i]:.04f}, {over_output_max[i]:.04f}]\t=>\t[{best_interm_min[i]:.04f}, {best_interm_max[i]:.04f}]')
                    print(candidates[-2:])
                    print()
                    
            random.shuffle(candidates)
            self.tightening_candidates[subnet_idx] = candidates
        
        assert len(self.tightening_candidates[subnet_idx]) > 0
        candidates = self.tightening_candidates[subnet_idx][:batch]
        print(f'Extracted {len(candidates)=} from {len(self.tightening_candidates[subnet_idx])} candidates')
        self.tightening_candidates[subnet_idx] = self.tightening_candidates[subnet_idx][len(candidates):]
    
        return candidates
        
    def verify(self, objective, verify_batch, tighten_batch, timeout=3600, use_extra=True):
        self.start_time = time.time()
        self.reset()
        
        lb, _ = self._init_interm_bounds(objective, use_extra=use_extra)
        if lb >= 0.0:
            return 'unsat'
        
        self.iteration = 0
        while True:
            print('[+] Iteration:', self.iteration)
            self._under_estimate(
                subnet_idx=0,
                verify_batch=verify_batch,
                verify_timeout=10.0,
                tighten_batch=tighten_batch,
            )
            
            status = self._verify_subnet(
                subnet_idx=1, 
                objective=objective, 
                verify_batch=verify_batch,
                timeout=20.0,
            )
            
            if status not in ['timeout']:
                return status
            
            self.iteration += 1
            if time.time() - self.start_time > timeout:
                return status
        
        
        

def formatted_print(a, b, name):
    if a.numel() > 5:
        a = ', '.join([f'{_.item():.03f}' for _ in a.flatten()])
        b = ', '.join([f'{_.item():.03f}' for _ in b.flatten()])
    print(f'[{name}] lb:', a)
    print(f'[{name}] ub:', b)
    print()
    
    
if __name__ == "__main__":
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
    
    
    device = 'cuda'
    # device = 'cpu'
    split_idx = 3
    use_extra = True
    
    pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
    objective = dnf_objectives.pop(1)
    print(pytorch_model)
    
    if 0:
        verifier = DecompositionalVerifier(
            net=pytorch_model,
            input_shape=input_shape,
            min_layer=40,
            device=device,
        )    
        
        
        tic = time.time()
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
        
        
        
    verifier = DecompositionalVerifier(
        net=pytorch_model,
        input_shape=input_shape,
        min_layer=split_idx,
        device=device,
    )    
    
    tic = time.time()
    # lb, ub = verifier._init_interm_bounds(objective, use_extra=use_extra)
    # print(time.time() - tic)
    # assert torch.all(lb <= ub)
    # formatted_print(lb, ub, f'Split ({use_extra=})')
    
    status = verifier.verify(
        objective=objective,
        verify_batch=500, # batch size of sub-verifiers
        tighten_batch=50, # number of tightening candidates
        timeout=3600,
        use_extra=True,
    )
    
    
    print(status, time.time() - tic)