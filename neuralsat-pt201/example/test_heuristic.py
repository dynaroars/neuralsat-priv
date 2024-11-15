from verifier.interactive_verifier import InteractiveVerifier
from test import reset_settings, extract_instance
# from util.misc.logger import logger
# import logging

if __name__ == "__main__":
    
    ############## Preprocess ##############
    
    # logger.setLevel(logging.INFO)
    reset_settings()
    
    net_path = 'example/onnx/mnistfc-medium-net-151.onnx'
    vnnlib_path = 'example/vnnlib/prop_2_0.03.vnnlib'
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
        action, state = env.decide()
        obs, reward, done, info = env.step(action)
        print(len(env.domains_list), info)
    