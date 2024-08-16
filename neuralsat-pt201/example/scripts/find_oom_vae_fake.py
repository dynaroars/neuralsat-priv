import torch.nn as nn
import torch
import time

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule
from util.misc.torch_cuda_memory import gc_cuda

def build_layers(layer_configs, transpose=False):
    layers = []
    for config in layer_configs:
        if transpose:
            layers.append(nn.ConvTranspose2d(**config))
        else:
            layers.append(nn.Conv2d(**config))
        layers.append(nn.ReLU(inplace=True))
    return layers


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params = }')
    return total_params


class Unflatten(nn.Module):
    
    def __init__(self, start_dim, shape):
        super(Unflatten, self).__init__()
        self.shape = tuple(shape)
        self.start_dim = start_dim
        
    def forward(self, x):
        assert self.start_dim == 1
        return x.view(x.size(0), *self.shape)
    
    def __repr__(self):
        return f'Unflatten(start_dim={self.start_dim}, shape={self.shape})'

        
class CustomVAE(nn.Module):
    def __init__(self, input_shape, latent_dim, encoder_configs, decoder_configs):
        super(CustomVAE, self).__init__()
        
        encoder_layers = build_layers(encoder_configs, transpose=False)
        decoder_layers = build_layers(decoder_configs, transpose=True)

        with torch.no_grad():
            dummy = torch.randn(input_shape)
            enc_output = nn.Sequential(*encoder_layers)(dummy)
            enc_flattened_size = enc_output.flatten(1).size(1)
            enc_unflattened_size = enc_output.size()[1:]

        encoder_layers = encoder_layers + [nn.Flatten(1), nn.Linear(enc_flattened_size, latent_dim)]
        self.encoder = nn.Sequential(*encoder_layers)
        
        decoder_layers = [nn.Linear(latent_dim, enc_flattened_size), Unflatten(1, enc_unflattened_size), nn.ReLU(inplace=True)] + decoder_layers
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        recon_x = self.decoder(z)
        # y = self.classifier(recon_x)
        return recon_x


def non_optim_bounds(net, input_shape, lower, upper, c=None, device='cpu', method='crown-optimized', conv_mode='patches'):
    new_x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)
    extras = {'sparse_intermediate_bounds': False}
    print(f'Computing bounds using {method=} {c=} {conv_mode=} {extras=}')
    abstract = BoundedModule(
        model=net, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts={'conv_mode': conv_mode, 'verbosity': 0, **extras},
        # bound_opts={'conv_mode': 'matrix', 'verbosity': 0},
        device=device,
        verbose=False,
    )
    print(abstract(new_x))
    
    abstract.set_bound_opts({'optimize_bound_args': {'iteration': 2}})
    
    l, u = abstract.compute_bounds(
        x=(new_x,), 
        C=c,
        bound_upper=True, 
        method=method,
    )
    
    print(f'{l.shape=}')
    print(f'{l=}')
    print(f'{u=}')
    
    assert torch.all(l <= u)
    del abstract
    return l, u
    
    
if __name__ == "__main__":
        
    torch.manual_seed(0)
    
    enc_configs = [
        {'in_channels': 1, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 0},
        # {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0},
        {'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 0},
        {'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 0},
    ]
    
    dec_configs = [
        {'in_channels': 256, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0},
        {'in_channels': 128, 'out_channels': 64, 'kernel_size': 3, 'stride': 2, 'padding': 0, 'output_padding': 0},
        # {'in_channels': 32, 'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 0, 'output_padding': 0},
        {'in_channels': 64, 'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 0, 'output_padding': 0},
    ]

    d = 'cuda'
    size = 1
    shape = (1, 1, 8 * size, 8 * size)
    # shape = (1, 1, 28, 28)
    batch = 1
    latent_dim = 64
    
    c = torch.Tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, -1]]]).float()
    # c = None

    vae = CustomVAE(
        input_shape=shape,
        encoder_configs=enc_configs, 
        decoder_configs=dec_configs,
        latent_dim=latent_dim, 
    ).to(d)
    vae.eval()
    
    get_model_params(vae)
    
    eps = 1e-3
    il = torch.randn(batch, *shape[1:]).to(d)
    iu = il + eps
    
    y = vae(il)
    print(f'{il.shape=} {y.shape=}')
    exit()
    
    if 1:
        
        # el, eu = non_optim_bounds(vae, shape, il, iu, c=c, device=d, method='backward')
        # el, eu = non_optim_bounds(vae, shape, il, iu, c=c, device=d, method='crown-optimized', conv_mode='matrix')
        el, eu = non_optim_bounds(vae, shape, il, iu, c=c, device=d, method='backward', conv_mode='matrix')
        # print(f'{el=}')
        # print(f'{eu=}')
        time.sleep(10)
        exit()
        gc_cuda()
    
    if 1:
        il = torch.randn(batch, *shape[1:]).to(d)
        iu = il + .05
        
        el, eu = non_optim_bounds(vae.encoder, shape, il, iu, c=c, device=d)
        # non_optim_bounds(encoder, shape, l, u, d)
        print(f'{el.shape=}')
        gc_cuda()
        time.sleep(5)
    else:
        el = torch.randn(batch, latent_dim).to(d)
        eu = el + .05
        
    ol, ou = non_optim_bounds(vae.decoder, (shape[0], latent_dim), el, eu, c=c, device=d)
    gc_cuda()
    time.sleep(10)
    