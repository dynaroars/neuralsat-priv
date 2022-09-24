from pprint import pprint
import gurobipy as grb
import multiprocessing
import torch.nn as nn
import numpy as np
import contextlib
import random
import torch
import time
import copy
import re
import os

from heuristic.falsification import randomized_falsification
from heuristic.falsification import gradient_falsification
from dnn_solver.symbolic_network import SymbolicNetwork
from abstract.eran import deepzono, deeppoly
from lp_solver.lp_solver import LPSolver
from dnn_solver.worker import *

from utils.cache import BacksubCacher, AbstractionCacher
from utils.read_nnet import NetworkNNET
from utils.timer import Timers
from utils.misc import MP
import settings


class DNNTheoremProver:

    epsilon = 1e-6
    skip = 1e-3

    def __init__(self, net, spec, decider=None):
        self.net = net
        self.layers_mapping = net.layers_mapping
        self.spec = spec

        self.decider = decider

        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            self.model = grb.Model()
            self.model.setParam('OutputFlag', False)

        # input bounds
        bounds_init = self.spec.get_input_property()
        self.lbs_init = torch.tensor(bounds_init['lbs'], dtype=settings.DTYPE, device=net.device)
        self.ubs_init = torch.tensor(bounds_init['ubs'], dtype=settings.DTYPE, device=net.device)

        self.gurobi_vars = [
            self.model.addVar(name=f'x{i}', lb=self.lbs_init[i], ub=self.ubs_init[i]) 
            for i in range(self.net.n_input)
        ]

        self.count = 0 # debug

        self.solution = None

        # if settings.HEURISTIC_DEEPZONO:
        #     self.deepzono = deepzono.DeepZono(net)

        self.transformer = SymbolicNetwork(net)

        if settings.HEURISTIC_DEEPPOLY:
            self.flag_use_backsub = True
            for layer in net.layers:
                if isinstance(layer, nn.Conv2d):
                    self.flag_use_backsub = False
                    break
            if self.flag_use_backsub:
                self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=1000)
            else:
                self.deeppoly = deeppoly.DeepPoly(net, back_sub_steps=0)


        self.concrete = self.net.get_concrete((self.lbs_init + self.ubs_init) / 2.0)

        # clean trash
        # os.system('rm -rf gurobi/*')
        os.makedirs('gurobi', exist_ok=True)

        # test
        # self.decider.target_direction_list = [[self.rf.targets[0], self.rf.directions[0]]]

        self.last_assignment = {}        

        # pgd attack 
        self.backsub_cacher = BacksubCacher(self.layers_mapping, max_caches=10)
        self.mvars = grb.MVar(self.gurobi_vars)

        if net.n_input <= 10:
            self.implication_interval = 1
            self.flag_use_mvar = False
            Timers.tic('Randomized attack')
            self.rf = randomized_falsification.RandomizedFalsification(net, spec, seed=settings.SEED)
            stat, adv = self.rf.eval(timeout=1)
            print('Randomized attack:', stat)
            if stat=='violated':
                self.solution = adv[0]
            Timers.toc('Randomized attack')


        else:
            self.implication_interval = 10
            self.flag_use_mvar = True

            Timers.tic('PGD attack')
            self.gf = gradient_falsification.GradientFalsification(net, spec)
            stat, adv = self.gf.evaluate()
            print('PGD attack:', stat)
            if stat:
                assert spec.check_solution(net(adv))
                assert (adv >= self.gf.lower).all()
                assert (adv <= self.gf.upper).all()
                self.solution = adv
            Timers.toc('PGD attack')

        self.update_input_bounds_last_iter = False

        ###########################################################
        if True:
            print('- Use MVar:', self.flag_use_mvar)
            print('- Implication interval:', self.implication_interval)
            print()
        ###########################################################

        self.Q1 = multiprocessing.Queue()
        self.Q2 = multiprocessing.Queue()

        (lower, upper), hidden_bounds = self._compute_output_abstraction(self.lbs_init, self.ubs_init)
        if self.decider is not None and settings.DECISION != 'RANDOM':
            self.decider.update(output_bounds=(lower, upper), hidden_bounds=hidden_bounds)



    def _update_input_bounds(self, lbs, ubs):
        for i, var in enumerate(self.gurobi_vars):
            # if abs(lbs[i] - ubs[i]) < DNNTheoremProver.epsilon: # concretize
            var.lb = lbs[i]
            var.ub = ubs[i]

            # if (abs(var.lb - lbs[i]) > DNNTheoremProver.skip):
            #     var.lb = lbs[i]
            # if (abs(var.ub - ubs[i]) > DNNTheoremProver.skip):
            #     var.ub = ubs[i]
        self.model.update()
        self.update_input_bounds_last_iter = True
        return True


    def _restore_input_bounds(self):
        if self.update_input_bounds_last_iter:
            for i, var in enumerate(self.gurobi_vars):
                var.lb = self.lbs_init[i]
                var.ub = self.ubs_init[i]
            self.model.update()
            self.update_input_bounds_last_iter = False



    def _find_unassigned_nodes(self, assignment):
        assigned_nodes = list(assignment.keys()) 
        for k, v in self.layers_mapping.items():
            intersection_nodes = set(assigned_nodes).intersection(v)
            if len(intersection_nodes) == len(v):
                return_nodes = self.layers_mapping.get(k+1, None)
            else:
                return set(v).difference(intersection_nodes)
        return return_nodes

    def _get_equation(self, coeffs):
        # tic = time.time()
        expr = grb.LinExpr(coeffs[:-1], self.gurobi_vars) + coeffs[-1]
        # print(len(coeffs), time.time() - tic)
        return expr

    @torch.no_grad()
    def __call__(self, assignment):

        # debug
        self.count += 1
        cc = frozenset()

        if self.solution is not None:
            return True, {}, None
        # reset constraints
        # Timers.tic('Reset solver')
        # self._restore_input_bounds()
        # Timers.toc('Reset solver')

        Timers.tic('Find node')
        unassigned_nodes = self._find_unassigned_nodes(assignment)
        is_full_assignment = True if unassigned_nodes is None else False
        Timers.toc('Find node')

        # forward
        Timers.tic('backsub_dict')
        output_mat, backsub_dict = self.transformer(assignment)
        Timers.toc('backsub_dict')

        Timers.tic('Find caching assignment')
        cache_nodes = self.get_cache_assignment(assignment)
        remove_nodes = [n for n in self.last_assignment if n not in cache_nodes]
        new_nodes = [n for n in assignment if n not in cache_nodes]
        Timers.toc('Find caching assignment')

        flag_parallel_implication = False if unassigned_nodes is None else len(unassigned_nodes) > 50
        # flag_parallel_implication = True
        # print('flag_parallel_implication:', flag_parallel_implication)

        if not self.flag_use_mvar:
            Timers.tic('get cache backsub_dict')
            backsub_dict_expr = self.backsub_cacher.get_cache(assignment)
            Timers.toc('get cache backsub_dict')

            if len(remove_nodes):
                self.model.remove([self.model.getConstrByName(f'cstr[{node}]') for node in remove_nodes])


        if not self.flag_use_mvar:
            # convert to gurobi LinExpr
            Timers.tic('Get Linear Equation')
            # print(len(new_nodes), len(assignment), len(self.last_assignment))
            if backsub_dict_expr is not None:
                for node in new_nodes:
                    if node not in backsub_dict_expr:
                        backsub_dict_expr[node] = self._get_equation(backsub_dict[node])
            else:
                backsub_dict_expr = {k: self._get_equation(backsub_dict[k]) for k in new_nodes}
                self.backsub_cacher.put({k: assignment[k] for k in backsub_dict_expr}, backsub_dict_expr)

            Timers.toc('Get Linear Equation')


            # add constraints
            Timers.tic('Add constraints')
            if len(new_nodes):
                for node in new_nodes:
                    status = assignment.get(node, None)
                    # assert status is not None
                    if status:
                        ci = self.model.addLConstr(backsub_dict_expr[node] >= DNNTheoremProver.epsilon, name=f'cstr[{node}]')
                    else:
                        ci = self.model.addLConstr(backsub_dict_expr[node] <= 0, name=f'cstr[{node}]')
            Timers.toc('Add constraints')

        else:

            Timers.tic('Add constraints')
            if len(remove_nodes) == 0 and len(new_nodes) <= 2:
                # print(len(new_nodes), len(assignment), len(backsub_dict))
                # exit()
                for node in new_nodes:
                    status = assignment.get(node, None)
                    # assert status is not None
                    eqx = self._get_equation(backsub_dict[node])
                    if status:
                        self.model.addLConstr(eqx >= DNNTheoremProver.epsilon)
                    else:
                        self.model.addLConstr(eqx <= 0)

            elif len(assignment) > 0:
                lhs = np.zeros([len(backsub_dict), len(self.gurobi_vars)])
                rhs = np.zeros(len(backsub_dict))
                # mask = np.zeros(len(mat_dict), dtype=np.int32)
                for i, node in enumerate(backsub_dict):
                    status = assignment.get(node, None)
                    if status is None:
                        continue
                    # mask[i] = 1
                    if status:
                        lhs[i] = -1 * backsub_dict[node][:-1]
                        rhs[i] = backsub_dict[node][-1] - 1e-6
                    else:
                        lhs[i] = backsub_dict[node][:-1]
                        rhs[i] = -1 * backsub_dict[node][-1]

                self.model.remove(self.model.getConstrs())
                self.model.addConstr(lhs @ self.mvars <= rhs) 
            Timers.toc('Add constraints')

        self.model.update()

        
        # caching assignment
        self.last_assignment = assignment

        # upper objective
        # self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MAXIMIZE)

        # check satisfiability
        Timers.tic('Check output property')
        if not is_full_assignment:
            self._optimize()
            Timers.toc('Check output property')
            if self.model.status == grb.GRB.INFEASIBLE:
                # print('call from partial assignment')
                self._restore_input_bounds()
                return False, cc, None

            if settings.DEBUG:
                print('[+] Check partial assignment: `SAT`')


        else: # output
            flag_sat = False
            output_constraint = self.spec.get_output_property(
                [self._get_equation(output_mat[i]) for i in range(self.net.n_output)]
            )
            for cnf in output_constraint:
                ci = [self.model.addLConstr(_) for _ in cnf]
                self._optimize()
                self.model.remove(ci)
                if self.model.status == grb.GRB.OPTIMAL:
                    if self.check_solution(self.get_solution()):
                        flag_sat = True
                        break

            Timers.toc('Check output property')
            if flag_sat:
                self.solution = self.get_solution()
                return True, {}, is_full_assignment
            # print('call from full assignment')
            self._restore_input_bounds()
            return False, cc, None


        # compute new input lower/upper bounds
        # Timers.tic('Tighten bounds')
        # # upper
        # if self.model.status == grb.GRB.OPTIMAL:
        #     ubs_lom = torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device)
        # else:
        #     ubs_lom = self.ubs_init

        # # lower
        # self.model.setObjective(grb.quicksum(self.gurobi_vars), grb.GRB.MINIMIZE)
        # self._optimize()
        # if self.model.status == grb.GRB.OPTIMAL:
        #     lbs_lom = torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device)
        # else:
        #     lbs_lom = self.lbs_init

        # Timers.toc('Tighten bounds')

        # if torch.any(lbs > ubs):
        #     print('hehe')
        #     print(max_val, min_val)
        #     should_update_input = False
        #     # return False, cc, None

        # if should_update_input:

        #     Timers.tic('Update bounds')
        #     stat = self._update_input_bounds(lbs, ubs)
        #     Timers.toc('Update bounds')
                
        #     if not stat: # conflict
        #         # print('call from update bounds')
        #         return False, cc, None


        # Timers.tic('Cache abstraction')
        # score = self.abstraction_cacher.get_score((lbs, ubs))
        # Timers.toc('Cache abstraction')

        # should_run_abstraction = True
        # print('should_run_abstraction:', should_run_abstraction)

        # if should_run_abstraction:
        # reachable heuristic:
        # if self.count % self.implication_interval == 1:
        if 1:
            tic = time.time()
            Timers.tic('Tighten bounds')
            flag_parallel_tighten = False

            lbs = self.lbs_init.clone()
            ubs = self.ubs_init.clone()

            if not flag_parallel_tighten:
                for idx, v in enumerate(self.gurobi_vars):
                    self.model.setObjective(v, grb.GRB.MINIMIZE)
                    self._optimize()
                    if self.model.status == grb.GRB.OPTIMAL:
                        lbs[idx] = v.X
                    else:
                        continue

                    self.model.setObjective(v, grb.GRB.MAXIMIZE)
                    self._optimize()
                    if self.model.status == grb.GRB.OPTIMAL:
                        ubs[idx] = v.X
            else:
                new_bounds = {}
                wloads = list(range(self.net.n_input))
                # random.shuffle(wloads)
                wloads = MP.get_workloads(wloads, n_cpus=os.cpu_count() // 2)
                workers = [multiprocessing.Process(target=tighten_bound_worker, 
                                                   args=(self.model.copy(), wl, self.Q1, f'Thread {i}', 20),
                                                   name=f'Thread {i}',
                                                   daemon=True) for i, wl in enumerate(wloads)]
                for w in workers:
                    w.start()

                for w in workers:
                    res = self.Q1.get()
                    new_bounds.update(res)

                for w in workers:
                    w.terminate()

                print(len(new_bounds))

                for n in new_bounds:
                    lbs[n] = new_bounds[n]['lbs']
                    ubs[n] = new_bounds[n]['ubs']

            Timers.toc('Tighten bounds')

            # print('old lbs:', self.lbs_init)
            # print('new lbs:', lbs)
            # print()
            # print('old ubs:', self.ubs_init)
            # print('new ubs:', ubs)
            # print('=========================>', time.time() - tic)
            # exit()

            # idx1 = torch.where(lbs_lom != lbs)
            # idx2 = torch.where(ubs_lom != ubs)
            # idx3 = torch.where(ubs_lom < lbs_lom)
            # print(len(idx1[0]), len(idx2[0]), len(idx3[0]))
            # print('ubs dung:', ubs[idx2])
            # print('ubs lom :', ubs_lom[idx2])


            # print('lbs dung:', lbs[idx1])
            # print('lbs lom :', lbs_lom[idx1])
            flag_cac = torch.all(lbs == self.lbs_init) and torch.all(ubs == self.ubs_init)
            print(flag_cac)
            # assert torch.all(lbs <= ubs)

            Timers.tic('Compute output abstraction')
            (lower, upper), hidden_bounds = self._compute_output_abstraction(lbs, ubs, assignment)
            Timers.toc('Compute output abstraction')


            Timers.tic('Heuristic Decision Update')
            if self.decider is not None and settings.DECISION != 'RANDOM':
                self.decider.update(output_bounds=(lower, upper), hidden_bounds=hidden_bounds)
            Timers.toc('Heuristic Decision Update')



            Timers.tic('Check output reachability')
            stat, should_run_again = self.spec.check_output_reachability(lower, upper)
            Timers.toc('Check output reachability')

            # self.abstraction_cacher.put((lbs, ubs), stat)
            # print(stat, score)

            if not stat: # conflict
                
                # Timers.tic('_single_range_check')
                # self._single_range_check(lbs, ubs, assignment)
                # Timers.toc('_single_range_check')

                print('call from reachability heuristic')
                self._restore_input_bounds()
                return False, cc, None

            if settings.HEURISTIC_DEEPPOLY and should_run_again and self.flag_use_backsub:
                Timers.tic('Deeppoly optimization reachability')
                Ml, Mu, bl, bu  = self.deeppoly.get_params()
                Timers.tic('Gen constraints')
                lbs_expr = [grb.LinExpr(wl, self.gurobi_vars) + cl for (wl, cl) in zip(Ml.detach().cpu().numpy(), bl.detach().cpu().numpy())]
                ubs_expr = [grb.LinExpr(wu, self.gurobi_vars) + cu for (wu, cu) in zip(Mu.detach().cpu().numpy(), bu.detach().cpu().numpy())]
                Timers.toc('Gen constraints')

                Timers.tic('Get output constraints')
                dnf_contrs = self.spec.get_output_reachability_constraints(lbs_expr, ubs_expr)
                Timers.toc('Get output constraints')

                flag_sat = False
                for cnf, adv_obj in dnf_contrs:
                    Timers.tic('Add constraints + Solve')
                    ci = [self.model.addLConstr(_) for _ in cnf]
                    self.model.setObjective(adv_obj, grb.GRB.MINIMIZE)
                    self._optimize()
                    self.model.remove(ci)
                    Timers.toc('Add constraints + Solve')
                    if self.model.status == grb.GRB.OPTIMAL:
                        tmp_input = torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)
                        if self.check_solution(tmp_input):
                            self.solution = tmp_input
                            Timers.toc('Deeppoly optimization reachability')
                            # print('ngon')
                            return True, {}, None
                        self.concrete = self.net.get_concrete(tmp_input)

                        flag_sat = True
                        break

                if not flag_sat:
                    print('call from optimized reachability heuristic')
                    Timers.toc('Deeppoly optimization reachability')
                    self._restore_input_bounds()
                    return False, cc, None
                Timers.toc('Deeppoly optimization reachability')

            if not self.flag_use_backsub:
                tmp_input = torch.tensor(
                    [random.uniform(lbs[i], ubs[i]) for i in range(self.net.n_input)], 
                    dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)

                if self.check_solution(tmp_input):
                    self.solution = tmp_input
                    return True, {}, None
                self.concrete = self.net.get_concrete(tmp_input)


        Timers.tic('Implications')
        implications = {}

        if self.count % self.implication_interval == 1:
            # parallel implication
            if flag_parallel_implication:
                backsub_dict_np = {k: v.detach().cpu().numpy() for k, v in backsub_dict.items()}
                wloads = MP.get_workloads(unassigned_nodes, n_cpus=os.cpu_count() // 2)
                # Q = multiprocessing.Queue()
                workers = [multiprocessing.Process(target=implication_worker, 
                                                        args=(self.model.copy(), backsub_dict_np, wl, self.Q2, self.concrete, f'Thread {i}'),
                                                        name=f'Thread {i}',
                                                        daemon=True) for i, wl in enumerate(wloads)]

                for w in workers:
                    w.start()

                for w in workers:
                    res = self.Q2.get()
                    implications.update(res)

                for w in workers:
                    w.terminate()

            else:
                if not self.flag_use_mvar:
                    # convert to gurobi LinExpr
                    Timers.tic('Get Linear Equation')
                    if backsub_dict_expr is not None:
                        backsub_dict_expr.update({k: self._get_equation(v) for k, v in backsub_dict.items() if k not in backsub_dict_expr})
                    else:
                        backsub_dict_expr = {k: self._get_equation(v) for k, v in backsub_dict.items()}

                    self.backsub_cacher.put(assignment, backsub_dict_expr)
                    Timers.toc('Get Linear Equation')


                # self._restore_input_bounds()
                for node in unassigned_nodes:
                    implications[node] = {'pos': False, 'neg': False}
                    # neg
                    if not self.flag_use_mvar:
                        eqx = backsub_dict_expr[node]
                    else:
                        eqx = self._get_equation(backsub_dict[node])
                    if self.concrete[node] <= 0:
                        ci = self.model.addLConstr(eqx >= DNNTheoremProver.epsilon)
                        self._optimize()
                        if self.model.status == grb.GRB.INFEASIBLE:
                            implications[node]['neg'] = True
                    else:
                    # pos
                        ci = self.model.addLConstr(eqx <= 0)
                        self._optimize()
                        if self.model.status == grb.GRB.INFEASIBLE:
                            implications[node]['pos'] = True
                    
                    self.model.remove(ci)

        Timers.toc('Implications')

        return True, implications, is_full_assignment

    def _optimize(self):
        self.model.update()
        self.model.reset()
        self.model.optimize()


    def get_solution(self):
        if self.model.status == grb.GRB.LOADED:
            self._optimize()
        if self.model.status == grb.GRB.OPTIMAL:
            return torch.tensor([var.X for var in self.gurobi_vars], dtype=settings.DTYPE, device=self.net.device).view(self.net.input_shape)
        return None


    def check_solution(self, solution):
        if torch.any(solution < self.lbs_init.view(self.net.input_shape)) or torch.any(solution > self.ubs_init.view(self.net.input_shape)):
            return False
        if self.spec.check_solution(self.net(solution)):
            return True
        return False

    def _compute_output_abstraction(self, lbs, ubs, assignment=None):
        if settings.HEURISTIC_DEEPZONO: # eran deepzono
            return self.deepzono(lbs, ubs)
        return self.deeppoly(lbs, ubs, assignment=assignment)


    def _shorten_conflict_clause(self, assignment, run_flag):
        return frozenset()
        Timers.tic('shorten_conflict_clause')
        if run_flag:
            cc = self.cs.shorten_conflict_clause(assignment)
            # print('unsat moi vao day', len(assignment), len(cc))
        else:
            cc = frozenset()
        Timers.toc('shorten_conflict_clause')
        # print('assignment =', assignment)
        # print()
        # exit()
        return cc

    def get_cache_assignment(self, assignment):
        cache_nodes = []
        if len(self.last_assignment) == 0 or len(assignment) == 0:
            return cache_nodes

        for idx, variables in self.layers_mapping.items():
            a1 = {n: self.last_assignment.get(n, None) for n in variables}
            a2 = {n: assignment.get(n, None) for n in variables}
            if a1 == a2:
                tmp = [n for n in a1 if a1[n] is not None]
                cache_nodes += tmp
                if len(tmp) < len(a1):
                    break
            else:
                for n in variables:
                    if n in assignment and n in self.last_assignment and assignment[n]==self.last_assignment[n]:
                        cache_nodes.append(n)
                break
        return cache_nodes