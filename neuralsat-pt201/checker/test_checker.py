from example.scripts.test_function import extract_instance
import random
import json
import time
    
from .checker import ProofChecker

def testcase_0():
    net_path = 'example/backup/motivation_example_159.onnx'
    vnnlib_path = 'example/backup/motivation_example_159.vnnlib'
    proof_trees = {3: [[-4], [-2, 4], [2, 1, 4], [2, -1, 4]], 4: []}
    return net_path, vnnlib_path, proof_trees
      
def testcase_1():
    net_path = 'example/onnx/mnist-net_256x2.onnx'
    vnnlib_path = 'example/vnnlib/prop_1_0.03.vnnlib'
    proof_trees = json.load(open('example/proof_tree_1.json'))
    # proof_trees = {4: [[35, -185, -320, -359, -432, 387]]}
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
    random.seed(0)
    net_path, vnnlib_path, proof_trees = testcase_1()
    pytorch_model, input_shape, dnf_objectives = extract_instance(net_path, vnnlib_path)
    print(pytorch_model)
    # print(f'{input_shape =}')
    tic = time.time()
    proof_checker = ProofChecker(pytorch_model, input_shape, dnf_objectives, verbose=False) 
    is_proved = proof_checker.prove(
        proofs=proof_trees, 
        batch=64, 
        expand_factor=1.0, 
        timeout_per_proof=1000.0,
        timeout=1000,
    )
    print(f'{is_proved = }, {time.time() - tic}')
    # print(dnf_objectives.ids)
    
        