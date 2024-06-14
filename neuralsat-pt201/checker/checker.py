import gurobipy as grb
import multiprocessing
from tqdm import tqdm
import torch
import copy
import time
import os

from .data import ProofQueue, ProofReturnStatus
from .milp_solver import build_milp_solver

MULTIPROCESS_MODEL = None

ALLOWED_GUROBI_STATUS_CODES = [
    grb.GRB.OPTIMAL, 
    grb.GRB.INFEASIBLE, 
    grb.GRB.USER_OBJ_LIMIT, 
    grb.GRB.TIME_LIMIT
]

def _proof_worker_impl(candidate):
    can_node, can_queue, can_var_mapping, _ = candidate
    # print(f'[{len(can_queue) = }] Solving {can_node = }')
    can_model = MULTIPROCESS_MODEL.copy()
    assert can_model.ModelSense == grb.GRB.MINIMIZE
    assert can_model.Params.BestBdStop > 0
    can_model.update()
    
    # add split constraints
    for literal in can_node.history:
        assert literal != 0
        relu_name, pre_relu_name, neuron_idx = can_var_mapping[abs(literal)]
        pre_var = can_model.getVarByName(f"lay{pre_relu_name}_{neuron_idx}")
        relu_var = can_model.getVarByName(f"ReLU{relu_name}_{neuron_idx}")
        # print(f'\t- {pre_relu_name=}, {neuron_idx=}, {relu_name=}')
        # print(f'\t- {literal=} {pre_var=}, {relu_var=} {pre_var.lb=} {pre_var.ub=}')
        assert pre_var is not None
        if relu_var is None: # var is None if relu is stabilized
            assert pre_var.lb * pre_var.ub >= 0, print('[!] Missing constraints')
            if (literal < 0 and pre_var.lb > 0) or (literal > 0 and pre_var.ub <= 0):
                # always unsat
                return float('inf')
        else:
            if literal > 0: # active
                can_model.addConstr(pre_var == relu_var)
            else: # inactive
                relu_var.lb = 0
                relu_var.ub = 0
        # TODO: remove all other relu_var relevant constraints
    can_model.update()
    can_model.optimize()

    # if can_model.status == grb.GRB.INF_OR_UNBD: # FIXME: sometimes Gurobi returns status code 4
    #     can_model.setParam(grb.GRB.Param.DualReductions, 0)
    #     can_model.optimize()
        
    assert can_model.status in ALLOWED_GUROBI_STATUS_CODES, print(f'[!] Error: {can_model=} {can_model.status=} {can_node.history=}')
    if can_model.status == grb.GRB.USER_OBJ_LIMIT: # early stop
        return 1e-5
    if can_model.status == grb.GRB.INFEASIBLE: # infeasible
        return float('inf')
    if can_model.status == grb.GRB.TIME_LIMIT: # timeout
        return can_model.ObjBound
    return can_model.objval
    
    
def _proof_worker_node(candidate):
    node, queue, _, _ = candidate
    if node is None:
        return False
        
    if not len(queue):
        return False
    
    max_filtered_nodes = queue.get_possible_filtered_nodes(node)
    if not max_filtered_nodes:
        return False
    
    obj_val = _proof_worker_impl(candidate)
    is_solved = obj_val > 0
    if is_solved:
        queue.filter(node)
    return is_solved
    


def _proof_worker(candidate):
    solved_node = None
    if _proof_worker_node(candidate): # solved
        node, queue, var_mapping, expand_factor = candidate
        solved_node = node
        while True:
            if expand_factor <= 1:
                break
            node = node // expand_factor
            new_candidate = (node, queue, var_mapping, expand_factor)
            if not _proof_worker_node(new_candidate):
                break
            solved_node = node
    return solved_node
                

class ProofChecker:
    
    def __init__(self, net, input_shape, dnf_objectives, verbose=False) -> None:
        self.net = net
        self.dnf_objectives = copy.deepcopy(dnf_objectives)
        self.input_shape = input_shape
        self.verbose = verbose
        self.device = 'cpu'

    @property
    def var_mapping(self) -> dict:
        if not hasattr(self, '_var_mapping'):
            self._var_mapping = {}
            count = 1
            for (pre_act_name, layer_size), act_name in zip(self.pre_relu_names, self.relu_names):
                for nid in range(layer_size):
                    self._var_mapping[count] = (act_name, pre_act_name, nid)
                    count += 1
        return self._var_mapping
    
    def build_core_checker(self, objectives, timeout=15.0):
        # TODO: generalize for different input ranges
        input_lower = objectives.lower_bounds[0].view(*self.input_shape[1:])
        input_upper = objectives.upper_bounds[0].view(*self.input_shape[1:])
        c_to_use = objectives.cs.to(self.device)
        assert c_to_use.shape[1] == 1, print(f'Unsupported shape {c_to_use.shape=}')
        c_to_use = c_to_use.transpose(0, 1)

        solver, solver_vars, self.pre_relu_names, self.relu_names = build_milp_solver(
            net=self.net,
            input_lower=input_lower,
            input_upper=input_upper,
            c=c_to_use,
            timeout=timeout,
            name='NeuralSAT_proof_checker',
        )    
        # setup objective
        assert len(objectives.cs) == len(solver_vars[-1])
        self.objective_vars_list = [(c_, var.VarName) for (c_, var) in zip(objectives.cs, solver_vars[-1])]
        return solver
        
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
        objective_var = new_model.getVarByName(objective_var_name) - rhs_to_use
        new_model.setObjective(objective_var, grb.GRB.MINIMIZE)
        new_model.update()
        return new_model
    
    def prove(self, proofs, batch=1, expand_factor=2.0, timeout=3600.0, timeout_per_proof=10.0):
        start_time = time.time()
        global MULTIPROCESS_MODEL
        # print(f'{proofs = }')
        proof_objectives = copy.deepcopy(self.dnf_objectives)
        
        # step 1: build common model without specific objective
        core_solver_model = self.build_core_checker(self.dnf_objectives)
        core_solver_model.setParam('TimeLimit', timeout_per_proof)
        core_solver_model.setParam('OutputFlag', self.verbose)
        core_solver_model.update()
        
        # step 2: prove each objective
        while len(proof_objectives):
            # check timeout
            if time.time() - start_time > timeout:
                return ProofReturnStatus.TIMEOUT 
            
            # step 2.1: get current objective
            current_objective = proof_objectives.pop(1)
            shared_solver_model = self.set_objective(core_solver_model, current_objective)

            # step 2.2: get current proof
            current_id = current_objective.ids.item()
            current_proof_queue = ProofQueue(proofs=proofs.get(current_id, []))
            
            # step 2.3: update shared model
            MULTIPROCESS_MODEL = shared_solver_model
            
            # step 2.4: prove nodes
            progress_bar = tqdm(total=len(current_proof_queue), desc=f"Processing objective {current_id}")
            while len(current_proof_queue):
                if time.time() - start_time > timeout:
                    return ProofReturnStatus.TIMEOUT 
                
                nodes = current_proof_queue.get(batch)
                candidates = [(node, current_proof_queue, self.var_mapping, expand_factor) for node in nodes]
                max_worker = min(len(candidates), os.cpu_count())
                with multiprocessing.Pool(max_worker) as pool:
                    results = pool.map(_proof_worker, candidates, chunksize=1)
                # print('Solved nodes:', results, len(current_proof_queue))
                processed = len(current_proof_queue)
                for solved_node in results:
                    if solved_node is not None:
                        current_proof_queue.filter(solved_node)
                # print(f'\t- Remaining: {len(current_proof_queue)}')
                processed -= len(current_proof_queue)
                progress_bar.update(processed)
                
            # step 2.5: delete shared model
            MULTIPROCESS_MODEL = None
            
        return ProofReturnStatus.CERTIFIED # proved
        