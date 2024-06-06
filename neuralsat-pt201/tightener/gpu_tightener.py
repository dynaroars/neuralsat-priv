import torch
import tqdm
import copy
import os

from util.network.read_onnx import parse_onnx, decompose_pytorch

from .utils import optimize_dnn, optimize_dnn_2, filter_dnf_pairs, verify_dnf_pairs

class GPUTightener:
    
    def __init__(self, verifier, abstractor):
        self.orig_verifier = copy.deepcopy(verifier)
        self.orig_net = copy.deepcopy(verifier.net)

        self.pre_activations = {i: layer.inputs[0].name for (i, layer) in enumerate(abstractor.net.perturbed_optimizable_activations)}
        self.input_shape = verifier.input_shape
        self.device = verifier.device
        
        self._setup_sub_networks()
        self._setup_sub_verifiers()
        
    def _setup_sub_networks(self):
        split_idx = 2
        self.sub_networks = {}
        while True:
            prefix_onnx_byte, _ = decompose_pytorch(self.orig_net.cpu(), self.orig_verifier.input_shape, split_idx)
            if prefix_onnx_byte is None:
                return
            # next layer
            # parse subnet
            prefix, _, output_shape, _ = parse_onnx(prefix_onnx_byte)
            # flatten output
            if len(output_shape) > 2:
                prefix = torch.nn.Sequential(prefix, torch.nn.Flatten(1))
                
            self.sub_networks[split_idx - 1] = prefix
            print(f'{split_idx = }')
            print(f'{prefix = }')
            print(f'{output_shape = }')
            print()
            split_idx += 1
        
    def _setup_sub_verifiers(self):
        self.sub_verifiers = {}
        for layer_id, subnet in self.sub_networks.items():
            # print('====================')
            # print(layer_id, subnet)
            sub_verifier = copy.deepcopy(self.orig_verifier)
            sub_verifier.net = subnet
            sub_verifier._setup_restart_naive(0, None)
            # print(sub_verifier.abstractor)
            # print(sub_verifier.net)
            # print(sub_verifier.abstractor.net)
            # print(sub_verifier.decision)
            self.sub_verifiers[layer_id] = sub_verifier
            # print('====================')
            # print()
        
            
    @torch.no_grad()
    def _sampling_random(self, layer_idx, lower_bounds, upper_bounds, n_sample=1000):
        x = (upper_bounds - lower_bounds) * torch.rand(n_sample, *self.input_shape[1:], device=self.device) + lower_bounds
        _, outputs = self.orig_net(x, return_interm=True)
        bounds = (outputs[layer_idx].min(0).values, outputs[layer_idx].max(0).values)
        return bounds
    
    
    def _sampling_gradient(self, layer_idx, lower_bounds, upper_bounds):
        min_i = optimize_dnn(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=True)
        max_i = optimize_dnn(self.sub_networks[layer_idx], lower_bounds, upper_bounds, is_min=False)
        bounds = (min_i.flatten(), max_i.flatten())
        return bounds
        
        
    def _sampling_gradient_2(self, layer_idx, lower_bounds, upper_bounds):
        bounds = [None]
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
        return best_min, best_max
        
        
    
    def __call__(self, domain_list):
        print(f'{type(self.orig_verifier) = }')
        print(f'{self.orig_net = }')
        print(f'{self.input_shape = }')
        
        worst_domains = domain_list.pick_out_worst_domains(len(domain_list), device='cpu')     
        unified_lower_bounds = {k: v.min(dim=0).values.flatten() for k, v in worst_domains.lower_bounds.items()}
        unified_upper_bounds = {k: v.max(dim=0).values.flatten() for k, v in worst_domains.upper_bounds.items()}  
        
        print(unified_lower_bounds.keys())
        for k, v in unified_lower_bounds.items():
            print(k, v.shape)
        print()
        
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(worst_domains.input_lowers <= worst_domains.input_uppers)
            assert torch.allclose(worst_domains.input_lowers.mean(0), worst_domains.input_lowers[0]), \
                print(worst_domains.input_lowers.mean(0).sum(), worst_domains.input_lowers[0].sum())
            assert torch.allclose(worst_domains.input_uppers.mean(0), worst_domains.input_uppers[0]), \
                print(worst_domains.input_uppers.mean(0).sum(), worst_domains.input_uppers[0].sum())
                
        input_lower = worst_domains.input_lowers[0:1]
        input_upper = worst_domains.input_uppers[0:1]
        
        idx = 1
        
        # for idx, (i1, i2) in enumerate(zip(interm1, interm2)):
        print(f'Layer {idx}, {self.pre_activations[idx]}')
        lower_bounds_i = unified_lower_bounds[self.pre_activations[idx]]
        upper_bounds_i = unified_upper_bounds[self.pre_activations[idx]]
        
        best_interm_min, best_interm_max = self.sampling(
            layer_idx=idx,
            input_lower=input_lower,
            input_upper=input_upper
        )
        
        n_outputs = len(best_interm_min)
        positive_neurons, negative_neurons = [], []
        for i in range(n_outputs):
            if lower_bounds_i[i] * upper_bounds_i[i] < 0:
                best_min = best_interm_min[i]
                best_max = best_interm_max[i]
                # print(f'[{i}]\t[{lower_bounds_i[i]:.02f}, {upper_bounds_i[i]:.02f}]\t => \t[{best_min:.02f}, {best_max:.02f}]')
                if best_min * best_max >= 0:
                    if best_min > 0:
                        positive_neurons.append(i)
                    elif best_max < 0:
                        negative_neurons.append(i)
        
        print() 
        print(f'{positive_neurons = }')
        print(f'{negative_neurons = }')
        print(f'{len(positive_neurons+negative_neurons) = }')

        filtered_positive_neurons, filtered_negative_neurons = filter_dnf_pairs(
            model=self.sub_networks[idx],
            input_lower=input_lower,
            input_upper=input_upper,
            n_outputs=n_outputs,
            positive_neurons=positive_neurons,
            negative_neurons=negative_neurons,
            n_iterations=1,
            patient_limit=1,
            eps=0.0,
        )
            
        print(f'{filtered_positive_neurons = }')
        print(f'{filtered_negative_neurons = }')
        print(f'{len(filtered_positive_neurons+filtered_negative_neurons) = }')
        
    
        # for i in filtered_positive_neurons+filtered_negative_neurons:
        #     print(f'[{i}]\t[{lower_bounds_i[i]:.02f}, {upper_bounds_i[i]:.02f}]\t => \t[{best_interm_min[i]:.02f}, {best_interm_max[i]:.02f}]')
            
        verify_dnf_pairs(
            verifier=self.sub_verifiers[idx],
            model=self.sub_networks[idx],
            input_lower=input_lower,
            input_upper=input_upper,
            n_outputs=n_outputs,
            positive_neurons=filtered_positive_neurons,
            negative_neurons=filtered_negative_neurons,
            eps=0.0,
        )