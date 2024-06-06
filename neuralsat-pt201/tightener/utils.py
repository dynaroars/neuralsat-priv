import torch
import tqdm
import time

from attacker.pgd_attack.general import attack as pgd_attack
from verifier.objective import Objective, DnfObjectives
from util.misc.adam_clipping import AdamClipping
from util.misc.check import check_solution


def generate_simple_specs(dnf_pairs, n_outputs):
    """
    Generate VNNLIB-based specification in negation format
    
    [[(0, 0.5, 'gt')],
     [(1, -0.5, 'gt')],
     [(1, 0.5, 'lt')],
     [(2, 0.1, 'gt')],
     [(2, 1.1, 'lt')],
     [(3, 0.2, 'gt')],
     [(3, 1.2, 'lt')],
     [(4, -0.1, 'gt')],
     [(0, 1.5, 'lt')],
     [(4, 0.31, 'lt')]]
    
    is equivalent to 
    
    ; Output constraints:
    (assert (or
        (and (>= Y_0 0.5))
        (and (>= Y_1 -0.5))
        (and (<= Y_1 0.5))
        (and (>= Y_2 0.1))
        (and (<= Y_2 1.1))
        (and (>= Y_3 0.2))
        (and (<= Y_3 1.2))
        (and (>= Y_4 -0.1))
        (and (<= Y_0 1.5))
        (and (<= Y_4 0.31))
    ))
    """
    
    all_cs = []
    all_rhs = []
    for cnf_pairs in dnf_pairs:
        cs = []
        rhs = []
        for output_i, rhs_i, direction in cnf_pairs:
            assert direction in ['lt', 'gt']
            c = torch.zeros(n_outputs)
            r = torch.tensor(rhs_i)
            c[output_i] = 1. 
            if direction == 'gt':
                c *= -1.
                r *= -1.
            cs.append(c)
            rhs.append(r)
        all_cs.append(torch.stack(cs))
        all_rhs.append(torch.stack(rhs))
        
    lengths = [len(_) for _ in all_cs]
    if len(set(lengths)) == 1:
        return torch.stack(all_cs), torch.stack(all_rhs)
    return all_cs, all_rhs    
    
    

def falsify_dnf_pairs(model, input_lower, input_upper, n_outputs, positive_neurons, negative_neurons, eps=1e-5):
    batch_size = 10
    dnf_pairs = [[(i, eps, 'lt')] for i in positive_neurons] + [[(i, -eps, 'gt')] for i in negative_neurons]
    # print(f'{positive_neurons = }')
    # print(f'{negative_neurons = }')
    all_cs, all_rhs = generate_simple_specs(dnf_pairs=dnf_pairs, n_outputs=n_outputs)
    x_attack = (input_upper - input_lower) * torch.rand(input_lower.shape, device=input_upper.device) + input_lower
    
    attack_indices = []
    for batch_idx in range(0, len(all_cs), batch_size):
        new_cs = all_cs[batch_idx:batch_idx+batch_size]
        new_rhs = all_rhs[batch_idx:batch_idx+batch_size]
        data_min_attack = input_lower.unsqueeze(1).expand(-1, len(new_cs), *input_lower.shape[1:])
        data_max_attack = input_upper.unsqueeze(1).expand(-1, len(new_cs), *input_upper.shape[1:])
        # print(new_cs.shape, data_max_attack.shape, input_upper.shape)
        is_attacked, attack_images = pgd_attack(
            model=model,
            x=x_attack, 
            data_min=data_min_attack,
            data_max=data_max_attack,
            cs=new_cs,
            rhs=new_rhs,
            attack_iters=50, 
            num_restarts=20,
            timeout=10.0,
        )
        if is_attacked:
            with torch.no_grad():
                for restart_idx in range(attack_images.shape[1]): # restarts
                    for prop_idx in range(attack_images.shape[2]): # props
                        attack_index = dnf_pairs[batch_idx+prop_idx][0][0]
                        if attack_index in attack_indices:
                            continue
                        adv = attack_images[:, restart_idx, prop_idx]
                        if check_solution(
                            net=model, 
                            adv=adv, 
                            cs=new_cs[prop_idx], 
                            rhs=new_rhs[prop_idx], 
                            data_min=data_min_attack[:, prop_idx], 
                            data_max=data_max_attack[:, prop_idx]
                        ):
                            attack_indices.append(attack_index)
                            assert torch.all(adv >= input_lower)
                            assert torch.all(adv <= input_upper)
                            # print(f'\t{attack_index = }, {dnf_pairs[batch_idx+prop_idx]}, {model(adv).flatten()[attack_index]}')
    
    # print(f'{attack_indices = }')
    # print()
    # exit()
    return attack_indices


def filter_dnf_pairs(model, input_lower, input_upper, n_outputs, positive_neurons, negative_neurons, n_iterations=20, patient_limit=2, eps=1e-5):
    attack_indices = []    
    pbar = tqdm.tqdm(range(n_iterations), desc='Filtering DNF Pairs')   
    patient = patient_limit
    for _ in pbar:
        new_P = [_ for _ in positive_neurons if _ not in attack_indices]
        new_N = [_ for _ in negative_neurons if _ not in attack_indices]
        new_indices = falsify_dnf_pairs(
            model=model,
            input_lower=input_lower,
            input_upper=input_upper,
            n_outputs=n_outputs,
            positive_neurons=new_P,
            negative_neurons=new_N,
            eps=eps,
        )
        if not len(new_indices):
            patient -= 1
            if patient < 0:
                break
        else:
            # reset patient
            patient = patient_limit
            
        attack_indices += new_indices
        pbar.set_postfix(attacked=len(attack_indices), patient=patient)
    
    attack_indices = list(sorted(set(attack_indices)))
    print(f'{len(attack_indices)=}, {attack_indices=}')
    
    filtered_P = [_ for _ in positive_neurons if _ not in attack_indices]
    filtered_N = [_ for _ in negative_neurons if _ not in attack_indices]
    return filtered_P, filtered_N
    
def verify_dnf_pairs(verifier, model, input_lower, input_upper, n_outputs, positive_neurons, negative_neurons, eps=1e-5):
    print('####### Start running other verifier here #######')
    dnf_pairs = [[(i, eps, 'lt')] for i in positive_neurons] + [[(i, -eps, 'gt')] for i in negative_neurons]
    print(f'{positive_neurons = }')
    print(f'{negative_neurons = }')
    all_cs, all_rhs = generate_simple_specs(dnf_pairs=dnf_pairs, n_outputs=n_outputs)
    print(f'{all_cs.shape = }, {all_rhs.shape = }')
    print(f'{verifier.input_shape = }')
    
    # objective
    objectives = []
    for spec_idx in range(len(all_cs)):
        input_bounds = torch.stack([input_lower.flatten(), input_upper.flatten()], dim=1)
        objectives.append(Objective((input_bounds.numpy().tolist(), (all_cs[spec_idx].numpy(), all_rhs[spec_idx].numpy()))))
            
    dnf_objectives = DnfObjectives(
        objectives=objectives, 
        input_shape=verifier.input_shape, 
        is_nhwc=False,
    )
    
    print(f'{dnf_objectives.cs.shape = }, {dnf_objectives.rhs.shape = }')
    
    assert torch.equal(all_cs, dnf_objectives.cs)
    assert torch.equal(all_rhs, dnf_objectives.rhs)
    
    count = 0
    verified = 0
    start_time = time.time()
    while len(dnf_objectives):
        count += 1
        objective = dnf_objectives.pop(1)
        if count != 84:
            continue
            # torch.onnx.export(
            #     verifier.net,
            #     torch.zeros(verifier.input_shape),
            #     'example/onnx/prefix_error.onnx',
            #     verbose=False,
            #     opset_version=12,
            # )
        # print(f'{objective.cs = }, {objective.rhs = }')
        verifier.start_time = time.time()
        try:
            stat = verifier._verify_one(objective, preconditions={}, reference_bounds={}, timeout=10)
            print(f'{stat=}')
        except:
            raise
            continue
        else:
            if stat == 'unsat':
                verified += 1
        
        remain = len(dnf_objectives)
        print(f'{count=}, {remain=}, {verified=}')
        
    print(f'{time.time() - start_time}, {verified=}')
        
            
    #     exit()
    # print()
    # self.other.input_split = self.input_split
    # self.other.start_time = time.time()
    # # cac_ref_bounds = self.other._setup_restart(0, objective)
    # # print(f'{cac_ref_bounds=}')
    # # cac = self.other._verify_one(objective=objective, preconditions={}, reference_bounds={}, timeout=100)
    # # print(f'{cac=}')
    # print()
    # print('####### End running other verifier here #######')
    # print()
    # print()
        


def optimize_dnn(net, lower, upper, n_sample=50, n_iteration=50, is_min=True):
    assert torch.all(lower <= upper)
    lower_expand = lower.expand(n_sample, *[-1] * (lower.ndim - 1))
    upper_expand = upper.expand(n_sample, *[-1] * (upper.ndim - 1))
    X = (torch.empty_like(lower_expand).uniform_() * (upper_expand - lower_expand) + lower_expand).requires_grad_()

    lr = torch.max(upper_expand - lower_expand).item() / 8
    optimizer = torch.optim.Adam([X], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    Fs = []
    for _ in tqdm.tqdm(range(n_iteration), desc='Minimizing' if is_min else 'Maximizing'):
        inputs = torch.max(torch.min(X, upper), lower)
        outputs = net(inputs)
        Fs.append(outputs.detach())
        loss = outputs # torch.clamp(outputs, min=-1e-3) if is_min else torch.clamp(outputs, max=1e-3)
        loss = loss.sum() if is_min else -loss.sum()
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
    
    Fs = torch.vstack(Fs)
    # print(f'{Fs.shape = }')
    if is_min:
        return Fs.min(0).values
    return Fs.max(0).values      


def optimize_dnn_2(net, lower, upper, n_sample=50, n_iteration=50, is_min=True):
    assert torch.all(lower <= upper)
    lower_expand = lower.expand(n_sample, *[-1] * (lower.ndim - 1))
    upper_expand = upper.expand(n_sample, *[-1] * (upper.ndim - 1))
    X = (torch.empty_like(lower_expand).uniform_() * (upper_expand - lower_expand) + lower_expand)
    
    delta_lower_limit = lower_expand - X
    delta_upper_limit = upper_expand - X
    delta = (torch.empty_like(X).uniform_() * (delta_upper_limit - delta_lower_limit) + delta_lower_limit).requires_grad_()
        
    lr = torch.max(upper_expand - lower_expand).item() / 8
    opt = AdamClipping(params=[delta], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.99)
    Fs = []
    for _ in tqdm.tqdm(range(n_iteration), desc='Minimizing' if is_min else 'Maximizing'):
        inputs = torch.max(torch.min((X + delta), upper_expand), lower_expand)
        output = net(inputs)
        Fs.append(output.detach())
        loss = output # torch.clamp(output, min=-1e-5)
        loss = loss.sum() if is_min else -loss.sum()
        loss.backward()
        opt.step(clipping=True, lower_limit=delta_lower_limit, upper_limit=delta_upper_limit, sign=1)
        opt.zero_grad(set_to_none=True)
        scheduler.step()
        
    Fs = torch.vstack(Fs)
    # print(f'{Fs.shape = }')
    if is_min:
        return Fs.min(0).values
    return Fs.max(0).values  