import random
import torch

from verifier.verifier import Verifier 
from test import extract_instance
from setting import Settings

from train.models.cnn.cnn import conv_base
from train.models.resnet.resnet import resnet_toy

def main():
    torch.manual_seed(0)
    random.seed(0)
    
    Settings.setup(None)
    
    net_path = 'example/onnx/mnistfc-medium-net-151.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    
    # net_path = 'example/onnx/convBigRELU__PGD.onnx'
    net_path = 'example/onnx/cifar10_2_255_simplified.onnx'
    vnnlib_path = 'example/vnnlib/cifar10_spec_idx_95_eps_0.00784.vnnlib'
    
    device = 'cpu'
    method = 'crown-optimized'
    extra_opts = {}

    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    
    # model = conv_base()
    model = resnet_toy()
    model.eval()
    
    verifier = Verifier(
        net=model, 
        input_shape=input_shape, 
        batch=1000,
        device=device,
    )
    objective = objectives.pop(1)
    print(model)
    
    verifier._init_abstractor(method, objective, extra_opts=extra_opts)
    print(verifier.abstractor)
    
    share_alphas = [
        n.name for n in verifier.abstractor.net.get_enabled_opt_act()
    ]
    print(share_alphas)

    # share_alphas = ['/24', '/26']
    share_alphas = ['/input.8', '/input.12', '/input.20', '/input.24', '/input.28', '/input.32', '/input.36',]
    # share_alphas = []
    verifier.abstractor.initialize(objective, share_alphas=share_alphas)


if __name__ == "__main__":
    main()