import random
import torch
import tqdm
import copy
import os


from .utils import optimize_dnn, optimize_dnn_2, filter_dnf_pairs, verify_dnf_pairs
from util.network.read_onnx import parse_onnx, decompose_pytorch

from setting import Settings

def setup_gpu_tightener_settings():
    config = {
        'use_restart': Settings.use_restart,
        'use_mip_tightening': Settings.use_mip_tightening
    }
    Settings.use_restart = 0
    Settings.use_mip_tightening = 0
    return config

def reset_gpu_tightener_settings(config):
    for k, v in config.items():
        setattr(Settings, k, v)
    

class GPUTightener:
    
    def __init__(self, verifier, abstractor, batch=100):
        assert not hasattr(verifier, "other")
        self.orig_verifier = copy.deepcopy(verifier)
        self.orig_verifier.batch = batch
        self.orig_net = copy.deepcopy(verifier.net).to(verifier.device)

        self.pre_activations = {i: layer.inputs[0].name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.input_shape = verifier.input_shape
        self.device = verifier.device
        
        self._setup_sub_networks()
        self._setup_sub_verifiers()
        self.reset()
        
    def print_bounds(self):
        print('Best intermediate bounds so far:')
        for (layer_idx, bounds) in self.best_interm_bounds.items():
            print(f'\t- {layer_idx=} min={bounds[0].sum().item():.04f} max={bounds[1].sum().item():.04f}')
        print()
        
    def reset(self):
        self.best_interm_bounds = {}
        self.tightened_layers = []
        
        
    def _setup_sub_networks(self):
        split_idx = 1
        self.sub_networks = {}
        while True:
            prefix_onnx_byte, _ = decompose_pytorch(self.orig_net, self.orig_verifier.input_shape, split_idx + 1)
            if prefix_onnx_byte is None:
                return
            # parse subnet
            prefix, _, output_shape, _ = parse_onnx(prefix_onnx_byte)
            
            # flatten output
            if len(output_shape) > 2:
                prefix = torch.nn.Sequential(prefix, torch.nn.Flatten(1))
                
            self.sub_networks[split_idx] = prefix.to(self.device)
            print(f'{split_idx = }')
            print(f'{prefix = }')
            print(f'{output_shape = }')
            print()
            split_idx += 1
        
    def _setup_sub_verifiers(self):
        self.sub_verifiers = {}
        for layer_id, subnet in self.sub_networks.items():
            # print(layer_id, subnet)
            sub_verifier = copy.deepcopy(self.orig_verifier)
            sub_verifier.net = subnet
            sub_verifier._setup_restart_naive(0, None)
            self.sub_verifiers[layer_id] = sub_verifier

        
            
    @torch.no_grad()
    def _sampling_random(self, layer_idx, lower_bounds, upper_bounds, n_sample=1000):
        x = (upper_bounds - lower_bounds) * torch.rand(n_sample, *self.input_shape[1:], device=self.device) + lower_bounds
        net = self.orig_net.to(self.device)
        _, outputs = net(x, return_interm=True)
        bounds = (outputs[layer_idx].min(0).values.flatten(), outputs[layer_idx].max(0).values.flatten())
        return bounds
    
    
    def _sampling_gradient(self, layer_idx, lower_bounds, upper_bounds):
        min_i = optimize_dnn(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=True)
        max_i = optimize_dnn(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=False)
        bounds = (min_i.flatten(), max_i.flatten())
        return bounds
        
        
    def _sampling_gradient_2(self, layer_idx, lower_bounds, upper_bounds):
        min_i = optimize_dnn_2(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=True)
        max_i = optimize_dnn_2(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=False)
        bounds = (min_i.flatten(), max_i.flatten())
        return bounds
    
    
    def sampling(self, layer_idx, input_lower, input_upper):
        (interms_gradient_min, interms_gradient_max) = self._sampling_gradient(
            layer_idx=layer_idx,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
        )
        
        (interms_gradient_min_2, interms_gradient_max_2) = self._sampling_gradient_2(
            layer_idx=layer_idx,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
        )
        
        (interms_random_min, interms_random_max) = self._sampling_random(
            layer_idx=layer_idx,
            lower_bounds=input_lower,
            upper_bounds=input_upper,
        )
        
        best_min = torch.min(torch.stack([interms_random_min, interms_gradient_min, interms_gradient_min_2]), dim=0).values
        best_max = torch.max(torch.stack([interms_random_max, interms_gradient_max, interms_gradient_max_2]), dim=0).values
        return [best_min, best_max]
        
        
    def select_layer(self):
        for l_id, l_name in self.pre_activations.items():
            if (l_id != 0) and ((l_id, l_name) not in self.tightened_layers):
                return l_id, l_name
        return random.choice(self.tightened_layers)
        
        
    def __call__(self, domain_list):
        layer_idx, layer_name = self.select_layer()
        print(f'{layer_idx=}, {layer_name=}, {self.tightened_layers=}')
        if (layer_idx, layer_name) not in self.tightened_layers:
            self.tightened_layers.append((layer_idx, layer_name))
            # self.stabilize(domain_list, layer_idx) # TODO: remove
            
        self.stabilize(domain_list, layer_idx)
        
    
    @torch.no_grad()
    def _update_best_interm_bounds_by_samples(self, layer_idx, samples):
        if not len(samples):
            return
        assert layer_idx in self.best_interm_bounds    
        model = self.sub_networks[layer_idx]
        output = model(samples)
        min_output = output.amin(0)
        max_output = output.amax(0)
        
        self.best_interm_bounds[layer_idx][0] = torch.where(
            self.best_interm_bounds[layer_idx][0] < min_output, 
            self.best_interm_bounds[layer_idx][0], 
            min_output
        )
        
        self.best_interm_bounds[layer_idx][1] = torch.where(
            self.best_interm_bounds[layer_idx][1] > max_output, 
            self.best_interm_bounds[layer_idx][1], 
            max_output
        )
  
                
    def extract_tightening_neurons(self, layer_idx, lower_bounds, upper_bounds, eps=1e-6):
        # positive_neurons, negative_neurons = [], []
        assert eps >= 0
        candidates = []
        extended_candidates = []
        assert layer_idx in self.best_interm_bounds
        best_interm_min, best_interm_max = self.best_interm_bounds[layer_idx]
        # print(f'{best_interm_min.shape = }')
        for i in range(len(best_interm_min)):
            if lower_bounds[i] * upper_bounds[i] < 0: # unstable neurons
                best_min = best_interm_min[i].item()
                best_max = best_interm_max[i].item()
                # print(f'[{i}]\t[{lower_bounds[i]:.02f}, {upper_bounds[i]:.02f}]\t => \t[{best_min:.02f}, {best_max:.02f}]')
                if best_min * best_max >= 0:
                    if best_min >= 0: # positive neuron
                        candidates.append([(i, 0.0 + eps, 'lt')])
                    elif best_max <= 0: # negative neuron
                        candidates.append([(i, 0.0 - eps, 'gt')])
                else:
                    # extended_candidates.append([(i, best_min + eps, 'lt')])
                    extended_candidates.append([(i, (best_min + lower_bounds[i]) / 2 + eps, 'lt')])
                    # print(f'\t- Added {i=} {best_min=} {lower_bounds[i]=} {(best_min + lower_bounds[i]) / 2 =}')
                    
                    # extended_candidates.append([(i, best_max - eps, 'gt')])
                    extended_candidates.append([(i, (best_max + upper_bounds[i]) / 2 - eps, 'gt')])
                    # print(f'\t- Added {i=} {best_max=} {upper_bounds[i]=} {(best_max + upper_bounds[i]) / 2 =}')
                    
                    
        if len(extended_candidates):
            # print(extended_candidates)
            extended_candidates = sorted(extended_candidates, key=lambda x: abs(x[0][1]), reverse=False)#[:100]
            # print(extended_candidates)

        return candidates + extended_candidates
        
        
    def falsify_layer(self, layer_idx, input_lower, input_upper, lower_bounds, upper_bounds, batch=20, iteration=5):
        for _ in range(iteration):
            # step 1: find possible stabilized neurons
            candidate_neurons = self.extract_tightening_neurons(
                layer_idx=layer_idx,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
            )
            if not len(candidate_neurons):
                return
            
            # step 2: falsify intermediate properties
            attack_samples = []
            pbar = tqdm.tqdm(range(0, len(candidate_neurons), batch), desc=f'[{_}-th] Falsifying {layer_idx=} {len(candidate_neurons)=}')
            for batch_id in pbar:
                # print(f'{batch_id=} {candidate_neurons[batch_id:batch_id+batch]=}')
                _, adv = filter_dnf_pairs(
                    model=self.sub_networks[layer_idx],
                    input_lower=input_lower,
                    input_upper=input_upper,
                    n_outputs=len(lower_bounds),
                    candidate_neurons=candidate_neurons[batch_id:batch_id+batch],
                    n_iterations=10, # TODO: update
                    patient_limit=1,
                )
                if len(adv):
                    attack_samples.append(adv)

                pbar.set_postfix(attack_samples=len(attack_samples))
                
            if not len(attack_samples):
                return

            attack_samples = torch.vstack(attack_samples)
            # print(f'{attack_samples.shape = }')
            
            # step 3: update best sampling bounds to avoid select wrong neurons in the future
            self._update_best_interm_bounds_by_samples(
                layer_idx=layer_idx, 
                samples=attack_samples,
            )
            # self.print_bounds()
            
            if len(attack_samples) / len(candidate_neurons) < 0.1:
                return
        
        
    def stabilize(self, domain_list, layer_idx):
        print(f'Layer {layer_idx}, {self.pre_activations[layer_idx]}')
        setting_config = setup_gpu_tightener_settings()
        
        # step 1: get worst domains
        worst_domains = domain_list.pick_out_worst_domains(len(domain_list), device='cpu')     
        unified_lower_bounds = {k: v.min(dim=0).values.flatten() for k, v in worst_domains.lower_bounds.items()}
        unified_upper_bounds = {k: v.max(dim=0).values.flatten() for k, v in worst_domains.upper_bounds.items()}  

        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(worst_domains.input_lowers <= worst_domains.input_uppers)
            assert torch.allclose(worst_domains.input_lowers.mean(0), worst_domains.input_lowers[0]), \
                print(worst_domains.input_lowers.mean(0).sum(), worst_domains.input_lowers[0].sum())
            assert torch.allclose(worst_domains.input_uppers.mean(0), worst_domains.input_uppers[0]), \
                print(worst_domains.input_uppers.mean(0).sum(), worst_domains.input_uppers[0].sum())
        
        # TODO: generalize for different input intervals
        input_lower = worst_domains.input_lowers[0:1].to(self.device)
        input_upper = worst_domains.input_uppers[0:1].to(self.device)
        
        # step 2: build intermediate properties
        if layer_idx not in self.best_interm_bounds:
            self.best_interm_bounds[layer_idx] = self.sampling(
                layer_idx=layer_idx,
                input_lower=input_lower,
                input_upper=input_upper
            )
            
        # step 3: falsify intermediate properties
        self.falsify_layer(
            layer_idx=layer_idx,
            input_lower=input_lower,
            input_upper=input_upper,
            lower_bounds=unified_lower_bounds[self.pre_activations[layer_idx]],
            upper_bounds=unified_upper_bounds[self.pre_activations[layer_idx]],
            batch=100,
            iteration=5, # TODO: update
        )
        
        # step 4: find possible stabilized neurons
        candidate_neurons = self.extract_tightening_neurons(
            layer_idx=layer_idx,
            lower_bounds=unified_lower_bounds[self.pre_activations[layer_idx]],
            upper_bounds=unified_upper_bounds[self.pre_activations[layer_idx]],
        )

        print(f'{len(candidate_neurons) = }')
        if not len(candidate_neurons):
            reset_gpu_tightener_settings(setting_config)
            return
        
        # step 5: verify intermediate properties
        verified_candidates, attack_samples = verify_dnf_pairs(
            verifier=self.sub_verifiers[layer_idx],
            input_lower=input_lower,
            input_upper=input_upper,
            n_outputs=len(unified_lower_bounds[self.pre_activations[layer_idx]]),
            candidate_neurons=candidate_neurons,
            batch=20,
            timeout=Settings.gpu_tightening_timeout,
            eps=0.0,
        )
        
        # update best bounds to avoid select wrong neurons in the future
        self._update_best_interm_bounds_by_samples(
            layer_idx=layer_idx, 
            samples=attack_samples,
        )
        
        # revert settings
        reset_gpu_tightener_settings(setting_config)
        
        if not len(verified_candidates):
            return
        
        # step 5: update verified bounds
        unified_lower_bounds_refined = {k: v.clone() for k, v in unified_lower_bounds.items()}
        unified_upper_bounds_refined = {k: v.clone() for k, v in unified_upper_bounds.items()}
        
        for (neuron_idx, neuron_bound, neuron_direction) in verified_candidates:
            layer_name = self.pre_activations[layer_idx]
            # print(neuron_idx, neuron_bound, neuron_direction, layer_name)
            assert neuron_direction in ['lt', 'gt']
            if neuron_direction == 'lt': # positive neuron
                assert unified_lower_bounds_refined[layer_name][neuron_idx] <= neuron_bound, print(unified_lower_bounds_refined[layer_name][neuron_idx], neuron_bound)
                unified_lower_bounds_refined[layer_name][neuron_idx] = neuron_bound
            else: # negative neuron
                assert unified_upper_bounds_refined[layer_name][neuron_idx] >= neuron_bound, print(unified_upper_bounds_refined[layer_name][neuron_idx], neuron_bound)
                unified_upper_bounds_refined[layer_name][neuron_idx] = neuron_bound
                
            print(f'Tightened {layer_name=} ({neuron_idx=}):\t[{unified_lower_bounds[layer_name][neuron_idx]:.04f}, {unified_upper_bounds[layer_name][neuron_idx]:.04f}]\t=>\t[{unified_lower_bounds_refined[layer_name][neuron_idx]:.04f}, {unified_upper_bounds_refined[layer_name][neuron_idx]:.04f}]')

        
        if os.environ.get('NEURALSAT_ASSERT'):
            assert all([torch.all(unified_lower_bounds_refined[key] <= unified_upper_bounds_refined[key]) for key in unified_upper_bounds_refined])
                    
        # step 6: update domains bounds
        class TMP:
            pass        
        
        refined_domain = TMP()
        refined_domain.lower_bounds = unified_lower_bounds_refined
        refined_domain.upper_bounds = unified_upper_bounds_refined
        domain_list.update_refined_bounds(refined_domain)
        return