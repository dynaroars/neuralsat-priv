import torch

from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule
from util.network.read_onnx import parse_onnx
from checker.abstractor import DeepPoly


if __name__ == "__main__":
    
    # input_shape = (1, 1, 4, 7)
    net_path = 'example/onnx/model-thanh.onnx'
    device = 'cpu'
    net, input_shape, _, _ = parse_onnx(net_path)
    
    dp = DeepPoly(net, device=device)
    
    lower = torch.randn(input_shape).to(device)
    upper = lower + 0.0
    upper[0, 60:] = lower[0, 60:] + 1
    
    (l, u), _ = dp(lower, upper)
    
    print(f'{lower=}')
    print(f'{upper=}')
    
    cac_lower = net(lower)
    cac_upper = net(upper)
    print('###########')
    
    print(f'{l=}')
    print(f'{u=}')
    
    print('-----')
    print(f'{cac_lower=}')
    print(f'{cac_upper=}')
    assert torch.all(l <= cac_lower)
    assert torch.all(cac_lower <= u)
    assert torch.all(l <= cac_upper)
    assert torch.all(cac_upper <= u)
    print('-----')
    
    
    
    
    # for h in hs:
    #     print(h.shape)
    #     print(h)
    
    print('###########')
    
    lp = BoundedModule(
        model=net, 
        global_input=torch.zeros(input_shape, device=device),
        bound_opts={'conv_mode': 'patches', 'verbosity': 0},
        device=device,
        verbose=False,
    )
    
    new_x = BoundedTensor(lower, PerturbationLpNorm(x_L=lower, x_U=upper)).to(device)
    with torch.no_grad():
        l, u = lp.compute_bounds(x=(new_x,), bound_upper=True)
        l, u = l[0], u[0]
    
    print(f'{l=}')
    print(f'{u=}')