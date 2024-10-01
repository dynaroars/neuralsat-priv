import torch.nn as nn
import torch

class ReLUNet(nn.Module):
    
    def __init__(self):
        super(ReLUNet, self).__init__()
        
        self.linear1 = nn.Linear(2, 3)
        self.linear1.weight.data = nn.Parameter(torch.tensor([[1, 2], [3, 4], [5, 6]]).float())
        self.linear1.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
        self.linear2 = nn.Linear(3, 2)
        self.linear2.weight.data = nn.Parameter(torch.tensor([1, 2, 3, -4, -5, -6]).view(2, 3).float())
        self.linear2.bias.data = nn.Parameter(torch.tensor([2, 3]).float())
        
        self.linear3 = nn.Linear(2, 3)
        self.linear3.weight.data = nn.Parameter(torch.tensor([[1, 2], [-3, -4], [-5, -6]]).float())
        self.linear3.bias.data = nn.Parameter(torch.tensor([1, 2, 3]).float())
        
    def forward(self, x):
        x = self.linear1(x)
        x = x.relu()
        x = self.linear2(x)
        x = x.relu()
        x = self.linear3(x)
        return x
    
    
def get_hidden_bounds(self, device):
    lower_bounds, upper_bounds = {}, {}
    # print(list(set(self.layers_requiring_bounds + self.split_nodes)))
    for layer in list(set(self.layers_requiring_bounds + self.split_nodes)):
        lower_bounds[layer.name] = layer.lower.detach().to(device)
        upper_bounds[layer.name] = layer.upper.detach().to(device)

    return lower_bounds, upper_bounds


def get_As(self):
    lAs, uAs = {}, {}
    # print(list(set(self.layers_requiring_bounds + self.split_nodes)))
    # print(self.roots)
    list_nodes = [r for r in self.roots() if r.lA is not None] + self.get_splittable_activations()
    for node in list_nodes:
        lA = getattr(node, 'lA', None)
        if lA is None:
            continue
        lAs[node.name] = lA.clone().transpose(0, 1)
        
        uA = getattr(node, 'uA', None)
        if uA is None:
            continue
        uAs[node.name] = uA.clone().transpose(0, 1)

    return list_nodes, (lAs, uAs)

def test_1():
    from auto_LiRPA.perturbations import PerturbationLpNorm
    from auto_LiRPA import BoundedModule, BoundedTensor

    net = ReLUNet()
    x_L = torch.tensor([[-1.0, -2.0]])
    x_U = torch.tensor([[1.0, 2.0]])
    
    print(net)
    
    device = 'cpu'
    method = 'backward'
    method = 'crown-optimized'
    relu_option = 'adaptive'
    # relu_option = 'zero-lb'
    
    abstractor = BoundedModule(
        model=net, 
        global_input=torch.zeros_like(x_L, device=device),
        bound_opts={
            'relu': relu_option, 
            'conv_mode': 'matrix', 
        },
        device=device,
        verbose=False,
    )
    new_x = BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(device)

    abstractor.eval()
    abstractor(new_x)
    abstractor.get_split_nodes()
    
    print(abstractor)
    lb, ub = abstractor.compute_bounds(x=(new_x,), method=method, C=None, bound_upper=True)
    print(f'[{method}] lower', lb)
    print(f'[{method}] upper', ub)
        
    c1 = torch.tensor([1, 0, 0]).view(1, -1)
    c2 = torch.tensor([0, 1, 0]).view(1, -1)
    c3 = torch.tensor([0, 0, 1]).view(1, -1)
    cs = torch.stack([c1, c3], dim=1)
    print(cs.shape)
    
    lb, ub = abstractor.compute_bounds(x=(new_x,), method=method, C=cs, bound_upper=True)
    print(f'[{method}] lower', lb)
    print(f'[{method}] upper', ub)
        
    # exit()
    
    if 0:
        print('###### After ######')
        lbs, ubs = get_hidden_bounds(abstractor, device)
        for i, k in enumerate(abstractor.split_nodes):
            print(f'\t- Layer {i+1}: lower={lbs[k.name]}')
            print(f'\t-        : upper={ubs[k.name]}')
            print()
            
        print('[backward] lower', lb)
        print('[backward] upper', ub)
        print()
        
    if 0:
        list_nodes, (lAs, uAs) = get_As(abstractor)
        for i, k in enumerate(list_nodes):
            print(f'- Layer {i+1}:', k)
            print('\t+ lA:')
            print(lAs[k.name])
            print()
            print('\t+ uA:')
            print(uAs[k.name])
            # print(f'\t{lAs[k.name]=}')
            # print(f'\t{uAs[k.name]=}')
            print()
    
if __name__ == "__main__":
    test_1()