import gurobipy as grb
import multiprocessing
import random
import torch
import copy
import json
import time
import os

from example.scripts.test_function import extract_instance
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule

MULTIPROCESS_MODEL = None

ALLOWED_GUROBI_STATUS_CODES = [
    grb.GRB.OPTIMAL, 
    grb.GRB.INFEASIBLE, 
    grb.GRB.USER_OBJ_LIMIT, 
    grb.GRB.TIME_LIMIT
]

def _proof_worker_impl(candidate):
    can_node, can_queue, can_var_mapping, can_activation_mapping, _ = candidate
    print(f'[{len(can_queue) = }] Solving {can_node = }')
    can_model = MULTIPROCESS_MODEL.copy()
    assert can_model.ModelSense == grb.GRB.MINIMIZE
    assert can_model.Params.BestBdStop > 0
    can_model.update()
    
    # add split constraints
    for literal in can_node.history:
        assert literal != 0
        pre_relu_name, neuron_idx = can_var_mapping[abs(literal)]
        relu_name = can_activation_mapping[pre_relu_name]
        # print(f'\t- {pre_relu_name=}, {neuron_idx=}, {relu_name=}')
        
        pre_var = can_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
        relu_var = can_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
        # print(f'\t- {pre_var=}, {relu_var=}')
        assert relu_var is not None # var is None if originally stable shouldn't appear here
        if literal > 0:
            # active
            can_model.addConstr(pre_var == relu_var)
        else:
            # inactive
            relu_var.lb = 0
            relu_var.ub = 0
        # TODO: remove all other relu_var relevant constraints
    can_model.update()
    can_model.optimize()

    # print(f'\t- obj_var =', can_model.getObjective().getVar(0))
    assert can_model.status in ALLOWED_GUROBI_STATUS_CODES, print(f'[!] Error: {can_model.status=}')
    if can_model.status == grb.GRB.USER_OBJ_LIMIT: # early stop
        return 1e-5
    if can_model.status == grb.GRB.INFEASIBLE: # infeasible
        return float('inf')
    if can_model.status == grb.GRB.TIME_LIMIT: # timeout
        return can_model.ObjBound
    return can_model.objval
    
    
            
def _proof_worker_node(candidate):
    node, queue, _, _, _ = candidate
    if node is None:
        return False
        
    if not len(queue):
        return False
    
    max_filtered_nodes = queue.get_possible_filtered_nodes(node)
    if not max_filtered_nodes:
        return False
    
    # print(f'[+] Solving {node=}, {max_filtered_nodes=}')
    obj_val = _proof_worker_impl(candidate)
    # print(f'\t- {obj_val = }')
    is_solved = obj_val > 0
    if is_solved:
        queue.filter(node)
        # print(f'\t- [{id(queue)}] Remaining {len(queue)}')
    return is_solved
    


def _proof_worker(candidate):
    solved_node = None
    if _proof_worker_node(candidate): # solved
        node, queue, var_mapping, activation_mapping, expand_factor = candidate
        solved_node = node
        while True:
            if expand_factor <= 1:
                break
            node = node // expand_factor
            new_candidate = (node, queue, var_mapping, activation_mapping, expand_factor)
            if not _proof_worker_node(new_candidate):
                break
            solved_node = node
    return solved_node
                
                
class Node:
    
    def __init__(self, history, name='node') -> None:
        self.history = history
        self.name = name
        
    def __len__(self):
        return len(self.history)
        
    def __lt__(self, other):
        "Compare solution spaces"
        if len(self) < len(other):
            return False
        for item in other.history:
            if item not in self.history:
                return False
        return True
    
    def __floordiv__(self, num):
        assert num >= 1
        if not len(self.history):
            return None
        return Node(history=self.history[:int(len(self)/num)], name=f'{self.name}_prefix')
    
    def __repr__(self):
        return f'Node({self.name}, {self.history})'


class ProofQueue:
    
    def __init__(self, proofs: list) -> None:
        histories = proofs if len(proofs) else [[]]
        self.queue = [Node(history=h, name=f'node_{i}') for i, h in enumerate(histories)]
        
    def get(self, batch):
        indices = random.sample(range(len(self)), min(len(self), batch))
        print(f'{batch=} {len(self)=} {indices=}')
        return [self.queue[idx] for idx in indices]
    
    def add(self, node: Node):
        self.queue.append(node)
    
    def filter(self, node: Node):
        "Filter out solved nodes"
        new_queue = [n for n in self.queue if not n < node]
        self.queue = new_queue
    
    def get_possible_filtered_nodes(self, node):
        if not len(node.history):
            return 1
        print(f'{node=}')
        nodes = [n for n in self.queue if n < node]
        return len(nodes)
    
    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        lists = []
        if len(self) > 10:
            lists += [str(n) for n in self.queue[:5]]
            lists += ['...']
            lists += [str(n) for n in self.queue[-5:]]
        else:
            lists += [str(n) for n in self.queue]
        # lists += [')']
        return '\nQueue(\n\t' + '\n\t'.join(lists) + '\n)'
            

class ProofChecker:
    
    def __init__(self, pytorch_model, input_shape, objectives, verbose=False) -> None:
        self.device = 'cpu'
        self.objectives = copy.deepcopy(objectives)
        self.input_shape = input_shape
        
        self.abstractor = BoundedModule(
            model=pytorch_model, 
            global_input=torch.zeros(input_shape, device=self.device),
            bound_opts={'conv_mode': 'matrix', 'verbosity': 0},
            device=self.device,
            verbose=False,
        )
        self.abstractor.eval()
        self.verbose = verbose

    
    @property
    def var_mapping(self) -> dict:
        if not hasattr(self, '_var_mapping'):
            self._var_mapping = {}
            count = 1
            for layer in self.abstractor.split_nodes:
                for nid in range(layer.lower.flatten(start_dim=1).shape[-1]):
                    self._var_mapping[count] = (layer.name, nid)
                    count += 1
        return self._var_mapping
    
    @property
    def activation_mapping(self) -> dict:
        if not hasattr(self, '_activation_mapping'):
            self._activation_mapping = {}
            for act in self.abstractor.splittable_activations:
                self._activation_mapping[act.inputs[0].name] = act.name
        return self._activation_mapping
    
    def new_input(self, x_L: torch.Tensor, x_U: torch.Tensor) -> BoundedTensor:
        if os.environ.get('NEURALSAT_ASSERT'):
            assert torch.all(x_L <= x_U)
        return BoundedTensor(x_L, PerturbationLpNorm(x_L=x_L, x_U=x_U)).to(self.device)
    
    
    def build_core_checker(self, objectives):
        self.abstractor._reset_solver_vars(self.abstractor.final_node())
        if hasattr(self.abstractor, 'model'): 
            del self.abstractor.model
        
        # MILP solver
        self.abstractor.model = grb.Model('NeuralSAT_proof_checker')
        # self.abstractor.model.setParam('Threads', 1)
        self.abstractor.model.setParam('OutputFlag', False)
        self.abstractor.model.setParam("FeasibilityTol", 1e-5)
        self.abstractor.model.setParam('BestBdStop', 1e-5) # Terminiate as long as we find a positive lower bound.
        self.abstractor.model.setParam('MIPGap', 1e-2)  # Relative gap between lower and upper objective bound 
        self.abstractor.model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between lower and upper objective bound 
        
        # TODO: generalize for different input ranges
        input_lower = objectives.lower_bounds[0].view(self.input_shape)
        input_upper = objectives.upper_bounds[0].view(self.input_shape)
        c_to_use = objectives.cs.to(self.device)
        assert c_to_use.shape[1] == 1, print(f'Unsupported shape {c_to_use.shape=}')
        c_to_use = c_to_use.transpose(0, 1)

        # compute perturbations
        new_x = self.new_input(input_lower, input_upper)
        self.abstractor.compute_bounds(x=(new_x,), C=c_to_use)
        self.abstractor.get_split_nodes()
        self.abstractor.build_solver_module(
            x=(new_x,), 
            C=c_to_use,
            final_node_name=self.abstractor.final_name, 
            model_type='mip', 
            timeout_per_neuron=10.0,
            refine=True,
        )
        self.abstractor.model.update()
    
        # setup objective
        output_vars = self.abstractor.final_node().solver_vars
        self.objective_vars_list = [
            (c_, var.VarName) for (c_, var) in zip(objectives.cs, output_vars)
        ]
        # print(f'{output_vars = }')
        # print(f'{self.objective_vars_list=}')
        return self.abstractor.model
        
    def set_objective(self, model, objective):
        new_model = model.copy()
        c_to_use = objective.cs[0]
        rhs_to_use = objective.rhs[0]
        
        objective_var_name = None
        for (c_, v_) in self.objective_vars_list:
            if torch.equal(c_, c_to_use):
                objective_var_name = v_
                break
        assert objective_var_name is not None
        # print(f'{c_to_use = }')
        # print(f'{objective_var_name = }')
        # print()
        objective_var = new_model.getVarByName(objective_var_name) - rhs_to_use
        new_model.setObjective(objective_var, grb.GRB.MINIMIZE)
        new_model.update()
        return new_model
    
    
    def prove(self, proofs, batch=1, expand_factor=2.0, timeout=3600.0, timeout_per_proof=10.0):
        start_time = time.time()
        global MULTIPROCESS_MODEL
        # print(f'{proofs = }')
        proof_objectives = copy.deepcopy(self.objectives)
        
        # step 1: build common model without specific objective
        core_solver_model = self.build_core_checker(self.objectives)
        core_solver_model.setParam('TimeLimit', timeout_per_proof)
        core_solver_model.setParam('OutputFlag', self.verbose)
        core_solver_model.update()
        
        # step 2: prove each objective
        while len(proof_objectives):
            # step 2.1: get current objective
            current_objective = proof_objectives.pop(1)
            shared_solver_model = self.set_objective(core_solver_model, current_objective)

            # step 2.2: get current proof
            current_id = current_objective.ids.item()
            current_proof_queue = ProofQueue(proofs=proofs.get(current_id, []))
            print(f'[{current_id=}] {len(current_proof_queue)=} {current_proof_queue}\n')
            
            # step 2.3: update shared model
            MULTIPROCESS_MODEL = shared_solver_model
            
            # step 2.4: prove nodes
            while len(current_proof_queue):
                nodes = current_proof_queue.get(batch)
                candidates = [(node, current_proof_queue, self.var_mapping, self.activation_mapping, expand_factor) for node in nodes]
                with multiprocessing.Pool(len(candidates)) as pool:
                    results = pool.map(_proof_worker, candidates, chunksize=1)
                # print('Solved nodes:', results, len(current_proof_queue))
                for solved_node in results:
                    if solved_node is not None:
                        current_proof_queue.filter(solved_node)
                print(f'\t- Remaining: {len(current_proof_queue)}')
                # exit()
                
            # step 2.5: delete shared model
            MULTIPROCESS_MODEL = None
            
            # step 2.6: check timeout
            if time.time() - start_time > timeout:
                return False # did not prove
            
        return True # proved
                
              
def testcase_0():
    net_path = 'example/backup/motivation_example_159.onnx'
    vnnlib_path = 'example/backup/motivation_example_159.vnnlib'
    proof_trees = {3: [[-4], [-2, 4], [2, 1, 4], [2, -1, 4]], 4: []}
    return net_path, vnnlib_path, proof_trees
      
def testcase_1():
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_1.json'))
    formatted_proof_trees = {int(k): v for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees


def testcase_1_direct():
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_1.json'))
    formatted_proof_trees = {int(k): [[]] for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees


def testcase_2():
    net_path = 'example/onnx/mnistfc-medium-net-151.onnx'
    vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_2.json'))
    formatted_proof_trees = {int(k): v for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees
    
    

def testcase_2_direct():
    net_path = 'example/onnx/mnistfc-medium-net-151.onnx'
    vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_2.json'))
    formatted_proof_trees = {int(k): [[]] for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees
    
    
def testcase_3():
    net_path = 'example/onnx/mnist-net_256x4.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_3.json'))
    formatted_proof_trees = {int(k): v for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees
    
def testcase_3_direct():
    net_path = 'example/onnx/mnist-net_256x4.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_3.json'))
    formatted_proof_trees = {int(k): [[]] for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees
    
if __name__ == "__main__":
    if 1:
        random.seed(0)
        net_path, vnnlib_path, proof_trees = testcase_3()
        pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
        print(pytorch_model)
        print(f'{input_shape =}')
        tic = time.time()
        proof_checker = ProofChecker(pytorch_model, input_shape, dnf_objectives, verbose=False) 
        is_proved = proof_checker.prove(
            proofs=proof_trees, 
            batch=32, 
            expand_factor=1.0, 
            timeout_per_proof=1000.0,
            timeout=1000,
        )
        print(f'{is_proved = }, {time.time() - tic}')
        # print(dnf_objectives.ids)
    else:
        node_1 = Node(name='aaaa', history=[2])
        node_2 = Node(name='bbbb', history=[1, 2, -4])
        
        print(f'{node_1 < node_2 = }')
        print(f'{node_2 < node_1 = }')
        
        queue = ProofQueue([[-4], [-2, 4], [2, 1, 4], [2, -1, 4]])
        print(queue)
        queue.add(node_1)
        queue.add(node_2)
        print(queue)
        queue.filter(node_1)
        print(queue)
        
        node_3 = node_2 // 2
        print(f'{node_3 = }')
        node_4 = node_3 // 2
        print(f'{node_4 = }')
        node_5 = node_4 // 2
        print(f'{node_5 = }')
        
        