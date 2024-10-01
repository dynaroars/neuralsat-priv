import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import argparse
import torch
import sys
import os

from util.spec.write_vnnlib import write_vnnlib
from train.models.vit.vit import *
from train.models.resnet.resnet import *


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


@torch.no_grad()
def test(test_loader, model, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout, disable=False)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X).argmax(-1)
        batch_acc = y.eq(y_pred).sum() / len(y)
        running_metric += batch_acc.item() * X.size(0)
    return running_metric / len(test_loader.dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--model_type', required=True)
    parser.add_argument('--benchmark_dir', default='example/generated_benchmark/')
    parser.add_argument('--data_root', default='train/data')
    parser.add_argument('--save_dir', default='train/weights')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--eps', type=float, default=0.01)
    parser.add_argument('--timeout', type=float, default=1000.0)

    args = parser.parse_args()
    # args.device = torch.device(args.device)
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    
    output_dir = os.path.join(args.benchmark_dir, args.model_type, f'eps_{args.eps:.06f}_{args.model_name}')
    spec_dir = os.path.join(output_dir, 'spec')
    net_dir = os.path.join(output_dir, 'net')
    csv_path = os.path.join(output_dir, 'instances.csv')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(net_dir, exist_ok=True)
    
    # dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    dataset = torchvision.datasets.CIFAR10(root=args.data_root, download=True, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    # network
    checkpoint_path = os.path.join(args.save_dir, args.model_type, f'{args.model_name}.pt')
    assert os.path.exists(checkpoint_path), f'{checkpoint_path=}'
    model = eval(args.model_name)()
    # model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model.to(args.device)
    model.eval()
    print(model)
    print(model(torch.ones(1, 3, 32, 32).to(args.device)))

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
        }
    )
    
    print('[+] Exporting Pytorch')
    torch.save(model, f"{net_dir}/{args.model_name}.pth")
    
    params = get_model_params(model)
    print(f'{params=}')
    
    
    
    # reload
    new_model = torch.load(f"{net_dir}/{args.model_name}.pth")
    acc = test(
        test_loader=dataloader, 
        model=new_model, 
        device=args.device,
    )

    print('Accuracy:', acc)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    pbar = tqdm(dataloader, desc=f'Generating specs for "{args.model_name}"')
    total, skip = 0, 0

    with open(csv_path, 'w') as fp:
        for i, (x, y) in enumerate(pbar):
            pbar.set_postfix(total=total, skip=skip)
            x = x.to(args.device)
            y = y.to(args.device)
            
            pred = new_model(x)
            # print(f'{pred=}')
            
            # skip incorrect prediction sample
            if pred.argmax(-1) != y: 
                skip += 1
                continue 
            
            data_lb = x - args.eps / 2
            data_ub = x + args.eps / 2
            spec_name = f'spec_idx_{i}_net_{args.model_name}_eps_{args.eps:.06f}_seed_{args.seed}.vnnlib'
            write_vnnlib(
                spec_path=f'{spec_dir}/{spec_name}',
                data_lb=data_lb,
                data_ub=data_ub,
                prediction=pred,
                negate_spec=True,
            )
            total += 1
        
            print(f'net/{args.model_name}.onnx,spec/{spec_name},{args.timeout}', file=fp)
            
        
if __name__ == '__main__':
    main()
