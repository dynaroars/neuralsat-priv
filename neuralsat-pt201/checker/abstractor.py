import torch.nn as nn
import torch

class DeepPoly:

    def __init__(self, net, device='cpu'):
        self.net = net.to(device)
        self.device = device

        self.transformers = self._build_network_transformer()

    def _build_network_transformer(self):
        last = InputTransformer().to(self.device)
        layers = [last]
        for layer in self.net.children():
            if isinstance(layer, nn.Linear):
                last = LinearTransformer(layer=layer, last=last)
            elif isinstance(layer, nn.ReLU):
                last = ReLUTransformer(last=last)
            elif isinstance(layer, nn.Flatten):
                last = FlattenTransformer(last=last)
            else:
                raise NotImplementedError(layer)
            layers += [last]
        return layers

    @torch.no_grad()
    def __call__(self, lower, upper):
        bounds = (lower, upper)
        hidden_bounds = {}
        for idx, transform in enumerate(self.transformers):
            bounds = transform(bounds)
            if isinstance(transform, LinearTransformer):
                hidden_bounds[idx-1] = bounds.clone() # additional input layer
            # print(f'\nProcessing {layer=} {bounds.shape = }')
            assert torch.all(bounds[0] <= bounds[1]), transform

        return (bounds[0], bounds[1]), hidden_bounds


class InputTransformer(nn.Module):

    def __init__(self, last=None):
        super(InputTransformer, self).__init__()
        self.last = last

    def forward(self, bounds):
        self.bounds = torch.vstack([bounds[0], bounds[1]])
        return self.bounds
    
    def __repr__(self):
        return 'Input'


class LinearTransformer(nn.Module):

    def __init__(self, layer, last=None):
        super(LinearTransformer, self).__init__()
        self.last = last
        self.weight = layer.weight
        self.bias = layer.bias
        self.W_plus = torch.clamp(self.weight, min=0.)
        self.W_minus = torch.clamp(self.weight, max=0.)

    def forward(self, bounds):
        upper = self.W_plus @ bounds[1] + self.W_minus @ bounds[0]
        lower = self.W_plus @ bounds[0] + self.W_minus @ bounds[1]
        self.bounds = torch.stack([lower, upper], 0) + self.bias.view(1, -1)
        self.back_sub()
        return self.bounds
    
    def back_sub(self):
        new_bounds = self._back_sub()
        new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]
        
    def _back_sub(self, params=None):
        # print('_back_sub', self, '--->', self.last)
        if params is None:
            params = self.weight.data, self.weight.data, self.bias.data, self.bias.data

        Ml, Mu, bl, bu = params

        if self.last.last is not None:
            if self.last.beta is not None:
                Mlnew = torch.clamp(Ml, min=0) * self.last.beta + torch.clamp(Ml, max=0) * self.last.lmbda
                Munew = torch.clamp(Mu, min=0) * self.last.lmbda + torch.clamp(Mu, max=0) * self.last.beta
                blnew = bl + torch.clamp(Ml, max=0) @ self.last.mu
                bunew = bu + torch.clamp(Mu, min=0) @ self.last.mu
                return self.last._back_sub(params=(Mlnew, Munew, blnew, bunew))
            else:
                return self.last._back_sub(params=params)
        else:
            lower = torch.clamp(Ml, min=0) @ self.last.bounds[0] + torch.clamp(Ml, max=0) @ self.last.bounds[1] + bl
            upper = torch.clamp(Mu, min=0) @ self.last.bounds[1] + torch.clamp(Mu, max=0) @ self.last.bounds[0] + bu
            return torch.stack([lower, upper], 0)


    def __repr__(self):
        return 'Linear'

class ReLUTransformer(nn.Module):

    def __init__(self, last=None):
        super(ReLUTransformer, self).__init__()
        self.last = last
    
    def forward(self, bounds):
        device = bounds.device
        ind2 = bounds[0] >= 0 
        ind3 = (bounds[1] > 0) * (bounds[0] < 0) 
        ind4 = (bounds[1] > -bounds[0]) * ind3

        self.bounds = torch.zeros_like(bounds, device=device)
        self.bounds[1, ind3] = bounds[1, ind3]
        self.bounds[:, ind4] = bounds[:, ind4]
        self.lmbda = torch.zeros_like(bounds[1], device=device)
        self.beta = torch.zeros_like(bounds[1], device=device)
        self.mu = torch.zeros_like(bounds[1], device=device)
        self.lmbda[ind2] = torch.ones_like(self.lmbda[ind2], device=device)

        diff = bounds[1, ind3] - bounds[0, ind3] 
        self.lmbda[ind3] = torch.div(bounds[1, ind3], diff)
        self.beta[ind4] = torch.ones_like(self.beta[ind4])
        self.mu[ind3] = torch.div(-bounds[0, ind3] * bounds[1, ind3], diff)
        self.bounds[:, ind2] = bounds[:, ind2]
        self.beta[ind2] = torch.ones_like(self.beta[ind2], device=device)

        self.back_sub()
        return self.bounds

    def back_sub(self):
        new_bounds = self._back_sub()
        new_bounds = new_bounds.reshape(self.bounds.shape)
        indl = new_bounds[0] > self.bounds[0]
        indu = new_bounds[1] < self.bounds[1]
        self.bounds[0, indl] = new_bounds[0, indl]
        self.bounds[1, indu] = new_bounds[1, indu]

    def _back_sub(self, params=None):
        # print('_back_sub', self, '--->', self.last)
        if params is None:
            device = self.beta.device
            params = torch.diag(self.beta), torch.diag(self.lmbda), torch.zeros_like(self.mu, device=device), self.mu
        Ml, Mu, bl, bu = params

        if self.last.last is not None:
            Mlnew = Ml @ self.last.weight
            Munew = Mu @ self.last.weight 
            blnew = bl + Ml @ self.last.bias
            bunew = bu + Mu @ self.last.bias
            return self.last._back_sub(params=(Mlnew, Munew, blnew, bunew))
        else:
            lower = torch.clamp(Ml, min=0) @ self.last.bounds[0] + torch.clamp(Ml, max=0) @ self.last.bounds[1] + bl
            upper = torch.clamp(Mu, min=0) @ self.last.bounds[1] + torch.clamp(Mu, max=0) @ self.last.bounds[0] + bu
            return torch.stack([lower, upper], 0)

    def __repr__(self):
        return 'ReLU'

class FlattenTransformer(nn.Module):
    
    def __init__(self, last=None):
        super(FlattenTransformer, self).__init__()
        self.last = last

    def forward(self, bounds):
        return bounds.flatten(1)
    
    def _back_sub(self, params):
        # print('_back_sub', self, '--->', self.last)
        Ml, Mu, bl, bu = params
        if self.last.last is not None:
            return self.last._back_sub(params=params)
        else:
            lower = (torch.clamp(Ml, min=0) @ self.last.bounds[0].flatten() + torch.clamp(Ml, max=0) @ self.last.bounds[1].flatten()).flatten() + bl
            upper = (torch.clamp(Mu, min=0) @ self.last.bounds[1].flatten() + torch.clamp(Mu, max=0) @ self.last.bounds[0].flatten()).flatten() + bu
            return torch.stack([lower, upper], 0)
    
    @property
    def beta(self):
        if hasattr(self.last, 'beta'):
            return self.last.beta.flatten()
        return None

    @property
    def mu(self):
        if hasattr(self.last, 'mu'):
            return self.last.mu.flatten()
        return None

    @property
    def lmbda(self):
        if hasattr(self.last, 'lmbda'):
            return self.last.lmbda.flatten()
        return None

    @property
    def bounds(self):
        return self.last.bounds

    def __repr__(self):
        return 'Flatten'

# Step 1: Define a simple model
class SimpleModel(nn.Module):
    
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 5)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
if __name__ == "__main__":
    
    net = SimpleModel()
    device = 'cuda'
    
    input_shape = (1, 1, 4, 7)
    abstractor = DeepPoly(net, device=device)
    
    lower = torch.randn(input_shape).to(device)
    upper = lower + 1
    
    (l, u), hs = abstractor(lower, upper)
    print(f'{l=}')
    print(f'{u=}')
    
    # for h in hs:
    #     print(h.shape)
    #     print(h)
        
    # from auto_LiRPA.perturbations import PerturbationLpNorm
    # from auto_LiRPA import BoundedTensor, BoundedModule
    
    # abstractor = BoundedModule(
    #     model=net, 
    #     global_input=torch.zeros(input_shape, device=device),
    #     bound_opts={'conv_mode': 'patches', 'verbosity': 0},
    #     device=device,
    #     verbose=False,
    # )
    
    # new_x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)
    # with torch.no_grad():
    #     l, u = abstractor.compute_bounds(x=(new_x,), bound_upper=True)
    #     l, u = l[0], u[0]
    
    # print(f'{l=}')
    # print(f'{u=}')