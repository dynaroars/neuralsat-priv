import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torchvision
import argparse
import torch
import sys
import os

loss_fn_p = torch.nn.CrossEntropyLoss()

def fgsm_attack(model, loss_fn, images, labels, epsilon):
    # Set requires_grad attribute of the images tensor
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass to calculate gradients
    loss.backward()
    
    # Collect the sign of the gradients
    sign_data_grad = images.grad.data.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = images + epsilon * sign_data_grad
    
    # Return the perturbed image
    return perturbed_image


def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def train(train_loader, model, criterion, optimizer, device='cpu', adv_train=False):
    """train function"""
    model.train()

    running_loss = 0.0
    pbar = tqdm(train_loader, desc='[Training]', file=sys.stdout)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)

        # clean
        optimizer.zero_grad()
        Y_rec = model(X)
        loss = criterion(Y_rec, y)
        loss.backward()
        optimizer.step()

        # adv
        if adv_train:
            optimizer.zero_grad()
            perturbed_images = fgsm_attack(model, loss_fn_p, X, y, epsilon=0.01)
            Y_perturb = model(perturbed_images)
            loss_p = loss_fn_p(Y_perturb, y)
            loss_p.backward()
            optimizer.step()
        
        
        running_loss += loss.item() * X.size(0)
        pbar.set_description(f'[Training iter {batch_id + 1}/{len(train_loader)}]'
                             f' batch_loss={loss.item():.03f}'
                            #  f' adv_loss={loss_p.item():.03f}'
                             )
    return running_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_loader, model, device='cpu'):
    """test function"""
    model.eval()

    running_metric = 0.0
    pbar = tqdm(test_loader, desc='[Testing]', file=sys.stdout)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)

        y_pred = model(X).argmax(-1)

        batch_acc = y.eq(y_pred).sum() / len(y)

        running_metric += batch_acc.item() * X.size(0)
        pbar.set_description(f'[Validation iter {batch_id + 1}/{len(test_loader)}]'
                             f' batch_acc={batch_acc.item():.03f}')
    return running_metric / len(test_loader.dataset)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', required=True)
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--save_dir', default='weights')
    parser.add_argument('--model', type=str, default='vit', choices=['vit', 'cvt'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--infer', action='store_true')

    args = parser.parse_args()
    args.device = torch.device(args.device)
    return args


def main():
    args = parse_args()
    
    if args.dataset == 'mnist':
        input_shape = (1, 1, 28, 28)
        dataset_class = torchvision.datasets.MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        input_shape = (1, 3, 32, 32)
        dataset_class = torchvision.datasets.CIFAR10
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ])
    
    train_set = dataset_class(
        root=args.data_root,
        train=True,
        transform=transform,
        download=True)
    
    test_set = dataset_class(
        root=args.data_root,
        train=False,
        transform=transform,
        download=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    if args.model == 'vit':
        from models.vit import get_model
        weights = args.infer
        model = get_model(
            input_shape=input_shape,
            depth=4, 
            num_heads=4, 
            patch_size=14, 
            embed_dim=256, 
            weights=weights,
        )
        if weights:
            model.to(args.device)
            val_epoch_acc = test(test_loader, model, args.device)
            print(f'val_acc={val_epoch_acc:.4f}')
            exit()
    elif args.model == 'cvt':
        from models.vit_3 import vit_2_4, vit_7_4
        from models.vit_4 import vit_7_4_32_sine
        model = vit_7_4_32_sine(dropout=0.1, attention_dropout=0.1)
    else:
        raise ValueError(args.model)
    model.to(args.device)
    print(model)

    params = get_model_params(model)
    print(f'{params=}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
    # warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5 / 5e-4, total_iters=10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch, eta_min=1e-6)
    # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[10])


    # train
    print('\n\n[Training]')
    for epoch in range(args.max_epoch):
        print(f'\n[Epoch {epoch + 1} / {args.max_epoch}] lr={scheduler.get_last_lr()[0]:.07f}')
        train_epoch_loss = train(train_loader, model, criterion, optimizer, args.device)
        print(f'[Epoch {epoch + 1} / {args.max_epoch}] train_loss={train_epoch_loss}')

        val_epoch_acc = test(test_loader, model, args.device)
        print(f'[Epoch {epoch + 1} / {args.max_epoch}] val_acc={val_epoch_acc:.4f}')
        scheduler.step()

    # save
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_file = os.path.join(args.save_dir, f'{args.output_name}.pt')
    torch.save(model.state_dict(), model_save_file)
    # torch.onnx.export(
    #     model.cpu(),
    #     torch.zeros(input_shape),
    #     os.path.join(args.save_dir, f'{args.output_name}.onnx'),
    #     verbose=False,
    #     opset_version=12,
    # )

if __name__ == '__main__':
    main()
