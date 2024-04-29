
import torch
import onnx

    
class IncompleteVerifier:
    
    def __init__(self, net, subnets, input_shape: tuple, device: str = 'cpu') -> None:
        self.net = net # pytorch model
        self.subnets = subnets # list of pytorch models
        self.device = device
        self.input_shape = input_shape
        
                
    @torch.no_grad()
    def _sampling(self, model, lower_bounds, upper_bounds, n_sample=100):
        x = (upper_bounds - lower_bounds) * torch.rand(n_sample, *lower_bounds.shape[1:], device=self.device) + lower_bounds
        # print(x)
        output = []
        for layer in model.children():
            if isinstance(layer, torch.nn.ReLU):
                output.append(x.clone())
            x = layer(x)
        return output
    
    def _verify_subnetwork(self, subnetwork, properties, timeout):
        "Verify intermediate properties"
        pass
    
    
    def _create_properties(self, bounds):
        "Create sample-based properties"
        pass
    
    def create_intermediate_property(self):
        "Create intermediate property"
        pass
            
            
    def backward_verify(self):
        "Verify in backward manner"
        pass
    
    
    def forward_verify(self):
        "Verify in forward manner"
        pass
    
    def mix_verify(self):
        pass
    
    def verify(self, mode):
        if mode == 'forward':
            return self.forward_verify()
        elif mode == 'backward':
            return self.backward_verify()
        return self.mix_verify()
    

if __name__ == "__main__":
    
    net_name = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_name = 'example/vnnlib/prop_1_0.05.vnnlib'

    net_name = 'example/onnx/motivation_example_3986.onnx'
    vnnlib_name = 'example/vnnlib/motivation_example_3986.vnnlib'
    