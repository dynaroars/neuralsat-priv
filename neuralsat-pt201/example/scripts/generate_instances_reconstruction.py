import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import onnxruntime as ort
from tqdm import tqdm
import numpy as np
import torchvision
import argparse
import random
import torch
import yaml
import onnx
import sys
import os

torch.use_deterministic_algorithms(True)

from train.models.vae.vae import VAE

from util.spec.write_vnnlib import write_vnnlib_recon_relation, write_vnnlib_recon_robust

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def inference_onnx(path: str, *inputs: np.ndarray):
    print('Infer:', path)
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    print(f'{names=}')
    return sess.run(None, dict(zip(names, inputs)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True, type=str)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--benchmark_dir', default='example/generated_benchmark/vae_robust/')
    parser.add_argument('--root_dir', default='train')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--eps', type=float, default=0.002)
    parser.add_argument('--timeout', type=float, default=1000.0)

    args = parser.parse_args()
    # args.device = torch.device(args.device)
    return args

def export_image(output_image, ngrid, output_folder, name, normalize=True):
    
    if normalize:
        output_image = torch.clamp(output_image, -1., 1.)
    output_image = (output_image + 1) / 2

    recon_grid = make_grid(output_image.cpu(), nrow=ngrid)
    recon_grid = torchvision.transforms.ToPILImage()(recon_grid)
    recon_grid.save(os.path.join(output_folder, f'samples_{name}.png'))
    
    
def choose_pairs(image, radius, n_pairs=1):
    assert len(image) == 1
    flatten_image = image.flatten()
    pairs = []
    while len(pairs) < n_pairs:
        # idx = random.randint(0, image.numel())
        indices = random.sample(range(flatten_image.numel()), 2)
        # print(f'{indices=}')
        if flatten_image[indices[0]] + 2 * radius < flatten_image[indices[1]]:
            pairs.append(tuple(indices)) 
        
    # print(pairs)    
    # exit()
    return pairs
        
        
        
@torch.no_grad()
def main_recon():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    config = yaml.safe_load(open(args.config))
    
    output_dir = os.path.join(args.benchmark_dir, args.model_name, f'eps_{args.eps:.06f}')
    spec_dir = os.path.join(output_dir, 'spec')
    net_dir = os.path.join(output_dir, 'net')
    csv_path = os.path.join(output_dir, 'instances.csv')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)
    
    # dataset
    train_config = config['train_params']
    if args.epoch:
        model_ckpt = os.path.join(args.root_dir, train_config['task_name'], train_config['vae_autoencoder_ckpt_name']) + f'_epoch_{args.epoch}.pth'
    else:
        model_ckpt = os.path.join(args.root_dir, train_config['task_name'], train_config['vae_autoencoder_ckpt_name']) + '.pth'
    assert os.path.exists(model_ckpt), f'{model_ckpt=}'
    
    dataset_class = torchvision.datasets.CIFAR10
    dataset_args = {'train': False, 'download': True}

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    dataset = dataset_class(
        root=os.path.join(args.root_dir, 'data'),
        transform=transform,
        **dataset_args
    )
    
    model = VAE(
        dataset_config=config['dataset_params'],
        model_config=config['autoencoder_params'],
    ).to(args.device)
    
    model.eval()
    print(f'Loading {model_ckpt=}')
    model.load_state_dict(torch.load(model_ckpt, map_location=args.device, weights_only=True))
    
    # export
    print('[+] Exporting ONNX')
    torch.onnx.export(
        model,
        torch.ones(1, 3, 32, 32).to(args.device),
        f'{net_dir}/{args.model_name}.onnx',
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        do_constant_folding=True
    )
    
    # from torch.onnx import verification

    # verification.verify(
    #     model.cuda(),
    #     torch.ones(1, 3, 32, 32).cuda(),
    #     opset_version=12,
    #     do_constant_folding=True,
    #     input_names=["input"],
    #     output_names=["output"],
    # )

    # verification.find_mismatch(
    #     model.cuda(), tuple(torch.ones(1, 3, 32, 32).cuda()), opset_version=12, do_constant_folding=True
    # )
    # exit()
    
    print('[+] Exporting Pytorch')
    model.eval()
    torch.save(model, f"{net_dir}/{args.model_name}.pth")
    del model
    new_model = torch.load(f"{net_dir}/{args.model_name}.pth").to(args.device)
    new_model.eval()
    
    # debug
    num_images = 64
    ngrid = 8
    idxs = torch.randint(0, len(dataset) - 1, (num_images,))
    ims = torch.cat([dataset[idx][0][None] for idx in idxs]).float()
    ims = ims.to(args.device)
    
    export_image(
        output_image=ims,
        ngrid=ngrid,
        output_folder=output_dir,
        name='input',
        normalize=False 
    )
    
    recon_onnx = torch.from_numpy(inference_onnx(
        f'{net_dir}/{args.model_name}.onnx', 
        ims.detach().cpu().numpy())[0]).to(args.device)
    export_image(
        output_image=recon_onnx,
        output_folder=output_dir,
        ngrid=ngrid,
        name='onnx',
        normalize=True 
    )
    
    
    recon_pytorch = new_model(ims)
    export_image(
        output_image=recon_pytorch,
        output_folder=output_dir,
        ngrid=ngrid,
        name='pytorch',
        normalize=True 
    )
    
    params = get_model_params(new_model)
    print(f'{params=}')
    
    diff = torch.norm(recon_onnx - recon_pytorch).item() / recon_pytorch.shape[0]
    print(f'{diff=}')
    diff_recon = torch.norm(ims - recon_pytorch).item() / recon_pytorch.shape[0]
    print(f'{diff_recon=}')
    assert diff < 1e-3
    # assert diff_recon < 20
    assert torch.allclose(recon_onnx, recon_pytorch, 1e-4, 1e-4)
    
    # reload
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    pbar = tqdm(dataloader, desc=f'Generating specs for "{args.model_name}"')
    total, skip = 0, 0

    with open(csv_path, 'w') as fp:
        for i, (x, y) in enumerate(pbar):
            pbar.set_postfix(total=total, skip=skip)
            x = x.to(args.device)
            y = new_model(x)
            diff = (x - y).abs().amax().item()
            # print(f'{i=} {diff=}')
            if diff > 0.3:
                skip += 1
                continue
            
            for j in range(1):
                # j = random.randint(0, x.numel())
                spec_name = f'spec_idx_{i}_{j}_net_{args.model_name}_eps_{args.eps:.06f}_seed_{args.seed}.vnnlib'
                write_vnnlib_recon_relation(
                    spec_path=f'{spec_dir}/{spec_name}',
                    center=x,
                    input_radius=args.eps,
                    select_pairs=choose_pairs(x, args.eps, n_pairs=1),
                    negate_spec=True,
                )
                total += 1
        
                print(f'net/{args.model_name}.onnx,spec/{spec_name},{args.timeout}', file=fp)
            
@torch.no_grad()
def main_robust():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    config = yaml.safe_load(open(args.config))
    
    output_dir = os.path.join(args.benchmark_dir, args.model_name, f'eps_{args.eps:.06f}')
    spec_dir = os.path.join(output_dir, 'spec')
    net_dir = os.path.join(output_dir, 'net')
    csv_path = os.path.join(output_dir, 'instances.csv')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)
    
    # dataset
    train_config = config['train_params']
    if args.epoch:
        model_ckpt = os.path.join(args.root_dir, train_config['task_name'], train_config['vae_autoencoder_ckpt_name']) + f'_epoch_{args.epoch}.pth'
    else:
        model_ckpt = os.path.join(args.root_dir, train_config['task_name'], train_config['vae_autoencoder_ckpt_name']) + '.pth'
    assert os.path.exists(model_ckpt), f'{model_ckpt=}'
    
    dataset_class = torchvision.datasets.CIFAR10
    dataset_args = {'train': False, 'download': True}

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    
    dataset = dataset_class(
        root=os.path.join(args.root_dir, 'data'),
        transform=transform,
        **dataset_args
    )
    
    model = VAE(
        dataset_config=config['dataset_params'],
        model_config=config['autoencoder_params'],
    ).to(args.device)
    
    model.eval()
    print(f'Loading {model_ckpt=}')
    model.load_state_dict(torch.load(model_ckpt, map_location=args.device, weights_only=True))
    
    # export
    print('[+] Exporting ONNX')
    torch.onnx.export(
        model,
        torch.ones(1, 3, 32, 32).to(args.device),
        f'{net_dir}/{args.model_name}.onnx',
        opset_version=12,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        do_constant_folding=True
    )
    
    print('[+] Exporting Pytorch')
    model.eval()
    torch.save(model, f"{net_dir}/{args.model_name}.pth")
    del model
    new_model = torch.load(f"{net_dir}/{args.model_name}.pth").to(args.device)
    new_model.eval()
    
    # debug
    num_images = 64
    ngrid = 8
    idxs = torch.randint(0, len(dataset) - 1, (num_images,))
    ims = torch.cat([dataset[idx][0][None] for idx in idxs]).float()
    ims = ims.to(args.device)
    
    export_image(
        output_image=ims,
        ngrid=ngrid,
        output_folder=output_dir,
        name='input',
        normalize=False 
    )
    
    recon_onnx = torch.from_numpy(inference_onnx(
        f'{net_dir}/{args.model_name}.onnx', 
        ims.detach().cpu().numpy())[0]).to(args.device)
    export_image(
        output_image=recon_onnx,
        output_folder=output_dir,
        ngrid=ngrid,
        name='onnx',
        normalize=True 
    )
    
    
    recon_pytorch = new_model(ims)
    export_image(
        output_image=recon_pytorch,
        output_folder=output_dir,
        ngrid=ngrid,
        name='pytorch',
        normalize=True 
    )
    
    params = get_model_params(new_model)
    print(f'{params=}')
    
    diff = torch.norm(recon_onnx - recon_pytorch).item() / recon_pytorch.shape[0]
    print(f'{diff=}')
    diff_recon = torch.norm(ims - recon_pytorch).item() / recon_pytorch.shape[0]
    print(f'{diff_recon=}')
    assert diff < 1e-3
    # assert diff_recon < 20
    assert torch.allclose(recon_onnx, recon_pytorch, 1e-4, 1e-4)
    
    # reload
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    pbar = tqdm(dataloader, desc=f'Generating specs for "{args.model_name}"')
    total, skip = 0, 0

    with open(csv_path, 'w') as fp:
        for i, (x, y) in enumerate(pbar):
            pbar.set_postfix(total=total, skip=skip)
            x = x.to(args.device)
            y = new_model(x)
            diff = (x - y).abs().amax().item()
            # print(f'{i=} {diff=}')
            if diff > 0.3:
                skip += 1
                continue
            
            for j in range(10):
                # j = random.randint(0, x.numel())
                spec_name = f'spec_idx_{i}_{j}_net_{args.model_name}_eps_{args.eps:.06f}_seed_{args.seed}.vnnlib'
                write_vnnlib_recon_robust(
                    spec_path=f'{spec_dir}/{spec_name}',
                    center=x,
                    input_radius=args.eps,
                    output_radius=5*args.eps,
                    num_out_prop=1,
                    seed=args.seed,
                    negate_spec=True,
                )
                total += 1
        
                print(f'net/{args.model_name}.onnx,spec/{spec_name},{args.timeout}', file=fp)
            
            
if __name__ == '__main__':
    main_robust()
