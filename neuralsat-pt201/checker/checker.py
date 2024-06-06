import gurobipy as grb
import multiprocessing
import torch
import copy
import json
import os

from example.scripts.test_function import extract_instance
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA import BoundedTensor, BoundedModule

MULTIPROCESS_MODEL = None

def _proof_worker(candidate):
    can_id, can_clause, can_var_mapping, can_activation_mapping = candidate
    print(f'[{can_id=}] {can_clause = }')
    # print(f'[{can_id=}] {can_var_mapping = }')
    # print(f'[{can_id=}] {can_activation_mapping = }')
    can_model = MULTIPROCESS_MODEL.copy()
    assert can_model.ModelSense == grb.GRB.MINIMIZE
    assert can_model.Params.BestBdStop > 0
    can_model.update()
    print(f'[{can_id=}] {id(can_model)=} {can_model=}')
    
    # add split constraints
    for literal in can_clause:
        assert literal != 0
        pre_relu_name, neuron_idx = can_var_mapping[abs(literal)]
        relu_name = can_activation_mapping[pre_relu_name]
        print(f'\t- {pre_relu_name=}, {neuron_idx=}, {relu_name=}')
        
        pre_var = can_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
        relu_var = can_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
        print(f'\t- {pre_var=}, {relu_var=}')
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
    print(f'[{can_id=}] {id(can_model)=} {can_model=}')
    can_model.optimize()
    
    assert can_model.status in [2, 15], print(f'[!] Error: {can_model.status=}')
    if can_model.status == 15:
        return 1e-5
    return can_model.objval
    

class ProofChecker:
    
    def __init__(self, pytorch_model, input_shape, objectives) -> None:
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

    
    @property
    def var_mapping(self) -> dict:
        if not hasattr(self, '_var_mapping'):
            self._var_mapping = {}
            count = 1
            for layer in self.abstractor.split_nodes:
                for nid in range(layer.lower.flatten(start_dim=1).shape[-1]):
                    self._var_mapping[count] = (layer.name, nid)
                    # self._var_mapping[layer.name, nid] = count
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
    
    
    def build_core_checker(self, objective):
        self.abstractor._reset_solver_vars(self.abstractor.final_node())
        if hasattr(self.abstractor, 'model'): 
            del self.abstractor.model
        
        # MILP solver
        self.abstractor.model = grb.Model('NeuralSAT_proof_checker')
        self.abstractor.model.setParam('OutputFlag', False)
        self.abstractor.model.setParam("FeasibilityTol", 1e-5)
        self.abstractor.model.setParam('BestBdStop', 1e-5) # Terminiate as long as we find a positive lower bound.
        # self.abstractor.model.setParam('MIPGap', 1e-2)  # Relative gap between lower and upper objective bound 
        # self.abstractor.model.setParam('MIPGapAbs', 1e-2)  # Absolute gap between lower and upper objective bound 
        
        # TODO: generalize for different input ranges
        input_lower = objective.lower_bounds.view(self.input_shape)
        input_upper = objective.upper_bounds.view(self.input_shape)
        c_to_use = objective.cs.to(self.device)
        rhs_to_use = objective.rhs[0].to(self.device)
        # print(f'{rhs_to_use = }')
        new_x = self.new_input(input_lower, input_upper)

        # compute perturbations
        self.abstractor.compute_bounds(x=(new_x,), C=c_to_use)
        self.abstractor.get_split_nodes()
        self.abstractor.build_solver_module(
            x=(new_x,), 
            C=c_to_use,
            final_node_name=self.abstractor.final_name, 
            model_type='mip', 
            refine=False,
        )
        self.abstractor.model.update()
        
        # setup objective
        output_vars = self.abstractor.final_node().solver_vars
        # print(f'{output_vars = }')
        # FIXME: support CNF constraints (more than one objectives)
        assert len(output_vars) == len(rhs_to_use) == 1
        objective_var = self.abstractor.model.getVarByName(output_vars[0].VarName) - rhs_to_use[0]
        # print(f'{objective_var = }')
        self.abstractor.model.setObjective(objective_var, grb.GRB.MINIMIZE)
        self.abstractor.model.update()
        return self.abstractor.model
        
        
    def prove(self, proofs):
        global MULTIPROCESS_MODEL
        # print(f'{proofs = }')
        proof_objectives = copy.deepcopy(self.objectives)
        while len(proof_objectives):
            # step 1: get current objective
            current_objective = proof_objectives.pop(1)
            check_core = self.build_core_checker(current_objective)

            # step 2: get current proof
            current_id = current_objective.ids.item()
            current_proof_tree = proofs[current_id]
            print(f'[{current_id=}] {current_proof_tree = }\n')
            
            # step3: create workers
            candidates = []
            for cidx, decisions in enumerate(current_proof_tree):
                candidates.append((cidx, decisions, self.var_mapping, self.activation_mapping))
            
            candidates = candidates[-1:] # TODO: remove
            
            if not len(candidates):
                candidates.append((0, [], self.var_mapping, self.activation_mapping))
                
            # step 4: run workers
            MULTIPROCESS_MODEL = check_core
            max_worker = min(len(candidates), os.cpu_count())
            max_worker = 1
            with multiprocessing.Pool(max_worker) as pool:
                results = pool.map(_proof_worker, candidates, chunksize=1)
            MULTIPROCESS_MODEL = None
            print(results)
            if any([_ < 0 for _ in results]):
                return False # disproved
            break # TODO: remove
                
        return True # proved
                
                
def testcase_1():
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree.json'))
    formatted_proof_trees = {int(k): v for k, v in proof_trees.items()}
    return net_path, vnnlib_path, formatted_proof_trees


def testcase_2():
    net_path = 'example/backup/motivation_example_159.onnx'
    vnnlib_path = 'example/backup/motivation_example_159.vnnlib'
    proof_trees = {3: [[-4], [-2, 4], [2, 1, 4], [2, -1, 4]]}
    return net_path, vnnlib_path, proof_trees
    
    
if __name__ == "__main__":
    net_path, vnnlib_path, proof_trees = testcase_1()
    pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
    print(pytorch_model)
    print(f'{input_shape =}')
    proof_checker = ProofChecker(pytorch_model, input_shape, dnf_objectives) 
    is_proved = proof_checker.prove(proof_trees)
    print(f'{is_proved = }')
    # print(dnf_objectives.ids)