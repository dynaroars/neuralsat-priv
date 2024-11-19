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
    
    
    name_dict = {
        i: node.name for i, node in enumerate(env.abstractor.net.split_nodes)
    }
    
    print(f'{name_dict=}')

    while not done:
        observation, subproblems = env.get_observation(batch)
        action, _state = env.decide(observation, subproblems)
        reward, done, info = env.step(action)
        print(action[0])
        for i in range(len(observation)):
            for j in range(len(observation[i])):
                print(f'observation[{i=}][{j=}]={observation[i][j].shape}')
        exit()
