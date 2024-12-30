import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
import logging
import tqdm
import sys
import os

from example.scripts.test_function import extract_instance
from train.models.resnet.resnet import *
from train.models.vit.vit import *
from setting import Settings

from decomposer.dec_verifier import DecompositionalVerifier
from attacker.attacker import Attacker
from util.misc.logger import logger
from torch import tensor

@torch.no_grad()
def evaluate_model(model, device='cpu'):
    """test function"""
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465], 
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./train/data/', download=False, transform=transform, train=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    running_metric = 0.0
    pbar = tqdm.tqdm(dataloader, desc='[Testing]', file=sys.stdout, disable=False)
    for batch_id, (X, y) in enumerate(pbar):
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X).argmax(-1)
        batch_acc = y.eq(y_pred).sum() / len(y)
        running_metric += batch_acc.item() * X.size(0)
    return running_metric / len(dataloader.dataset)


def test_resnet():    
    
    torch.manual_seed(36)
    
    
    Settings.setup(None)
    print(Settings)
    device = 'cuda'
    model_name = 'resnet18'
    eps = 0.0002
    
    
    benchmark_dir = f'example/generated_benchmark/resnet/eps_{eps:.06f}_{model_name}'
    output_dir = f'{benchmark_dir}/result'
    os.makedirs(output_dir, exist_ok=True)
    
    instances = [l.split(',')[:-1] for l in open(f'{benchmark_dir}/instances.csv').read().strip().split('\n')]
    
    for idx, inst in enumerate(tqdm.tqdm(instances, desc=model_name)):
        # if idx != 2:
        #     continue
        output_file = f'{output_dir}/{idx}.res'
        if os.path.exists(output_file):
            continue
        net_path = f'{benchmark_dir}/{inst[0]}'
        vnnlib_path = f'{benchmark_dir}/{inst[1]}'
        pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
        pytorch_model = pytorch_model.to(device)
        print(f'Running {idx=}')
        
        if idx == 0:
            acc = evaluate_model(
                model=pytorch_model, 
                device=device,
            )
            print('Accuracy:', acc)
            assert acc > 0.8, f'{acc=}'
            
            
        verifier = DecompositionalVerifier(
            net=pytorch_model,
            input_shape=input_shape,
            min_layer=6,
            device=device,
        )    
        
        # objective = dnf_objectives.pop(1)
        try:
            status, lb = verifier.decompositional_verify(dnf_objectives, timeout=10000, batch=50, interm_batch=200)
            print(status, lb)
            # exit()
            with open(output_file, 'w') as fp:
                print(f'{lb},{status},{net_path},{vnnlib_path}', file=fp)
        except SystemExit:
            exit()
        except KeyboardInterrupt:
            exit()
        except:
            import traceback; traceback.print_exc()
            pass
        

def test_vae():    
    
    torch.manual_seed(36)
    
    Settings.setup(None)
    logger.setLevel(logging.DEBUG)
    
    print(Settings)
    device = 'cuda'
    model_name = 'cifar10_4_2'
    eps = 0.001
    benchmark_dir = f'example/generated_benchmark/vae_robust/{model_name}/eps_{eps:.06f}'
    # benchmark_dir = f'example/generated_benchmark/resnet_no_bn/eps_{eps:.06f}_{model_name}'
    output_dir = f'{benchmark_dir}/result'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    instances = [l.split(',')[:-1] for l in open(f'{benchmark_dir}/instances.csv').read().strip().split('\n')]
    
    for idx, inst in enumerate(tqdm.tqdm(instances, desc=model_name)):
        # if idx != 2:
            # continue
        if idx < 12123: 
            continue
        output_file = f'{output_dir}/{idx}.res'
        if os.path.exists(output_file):
            res = open(output_file).read()
            if 'sat' in res:
                continue
            
            # if ',timeout,' in res:
            #     continue
            
            if ',error,' in res:
                continue
            
            if ',unknown,' in res:
                val = eval(res.split(',unknown,')[0])
                if val.item() < -1.0 or val.item() > 1.0:
                    print(f'Skipping {val=}')
                    continue
        else:
            continue
            
        net_path = f'{benchmark_dir}/{inst[0]}'
        vnnlib_path = f'{benchmark_dir}/{inst[1]}'
        pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
        print(f'\n========\nRunning {idx=}/{len(instances)}\n========\n')
        
        pytorch_model = pytorch_model.to(device)
        attacker = Attacker(
            net=pytorch_model, 
            objective=dnf_objectives, 
            input_shape=input_shape, 
            device=device,
        )
        
        is_attacked, adv = attacker.run(timeout=10.0)
        if is_attacked:
            print(f'Attacked {idx=}')
            with open(output_file, 'w') as fp:
                print(f'{float("inf")},sat,{net_path},{vnnlib_path}', file=fp)
            continue
        
        verifier = DecompositionalVerifier(
            net=pytorch_model,
            input_shape=input_shape,
            min_layer=1,
            device=device,
        )    
        
        # objective = dnf_objectives.pop(1)
        try:
            status, lb = verifier.decompositional_verify(dnf_objectives, timeout=1200, batch=500)
            print(f'{lb=}')
            with open(output_file, 'w') as fp:
                print(f'{lb},{status},{net_path},{vnnlib_path}', file=fp)
        except SystemExit:
            exit()
        except KeyboardInterrupt:
            exit()
        except AssertionError:
            import traceback; traceback.print_exc()
            with open(output_file, 'w') as fp:
                print(f'-inf,error,{net_path},{vnnlib_path}', file=fp)
        except:
            import traceback; traceback.print_exc()
            pass
            # raise
             
if __name__ == "__main__":
    # test_resnet()
    test_vae()