from neuralsat.test import reset_settings, extract_instance
from neuralsat import InteractiveVerifier
# from util.misc.logger import logger
# import logging

if __name__ == "__main__":
    
    ############## Preprocess ##############
    
    # logger.setLevel(logging.INFO)
    reset_settings()
    
    net_path = 'src/neuralsat/example/onnx/mnistfc-medium-net-151.onnx'
    vnnlib_path = 'src/neuralsat/example/vnnlib/prop_2_0.03.vnnlib'
    device = 'cpu'
    batch = 100
    print(f'\n\nRunning test with {net_path=} {vnnlib_path=}')
    
    model, input_shape, objectives = extract_instance(net_path, vnnlib_path)
    model.to(device)
    
    env = InteractiveVerifier(
        net=model,
        input_shape=input_shape,
        batch=batch,
        device=device,
    )
    
    objectives, _ = env._preprocess(objectives, force_split=None)
    objective = objectives.pop(1)
    
    ############## Verify ##############
    

    done = env.init(
        objective=objective, 
        reference_bounds=None,
        preconditions={}, 
    )
    
    while not done:
        obs, subproblems = env.get_observation(batch)
        action, _state = env.decide(obs, subproblems)
        reward, done, info = env.step(action)
        print(info)
    