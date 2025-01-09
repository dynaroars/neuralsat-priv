import onnxruntime as ort
import numpy as np
import torch
import onnx
import os

def inference_onnx(path: str, *inputs: np.ndarray):
    print('[+] Infer ONNX:', path)
    sess = ort.InferenceSession(onnx.load(path).SerializeToString())
    names = [i.name for i in sess.get_inputs()]
    print(f'{names=}')
    return sess.run(None, dict(zip(names, inputs)))

BENCHMARK_TYPE = 'vae_relation'
BENCHMARK_NAME = 'cifar10_4_2'

ROOT_DIR = f'/home/hai/Hai/decomposition/benchmarks/{BENCHMARK_TYPE}/{BENCHMARK_NAME}/net'

if __name__ == "__main__":
    device = 'cpu'
    print('[+] Load Pytorch')
    model = torch.load(f"{ROOT_DIR}/{BENCHMARK_NAME}.pth").to(device)

    # print('[+] Exporting ONNX')
    # torch.onnx.export(
    #     model,
    #     torch.ones(1, 3, 32, 32).to(device),
    #     f'{ROOT_DIR}/{BENCHMARK_NAME}.onnx',
    #     opset_version=12,
    #     input_names=["input"],
    #     output_names=["output"],
    #     dynamic_axes={
    #         'input': {0: 'batch_size'},
    #         'output': {0: 'batch_size'},
    #     },
    # )
    
    
    # cmd = f'onnxsim "{ROOT_DIR}/{BENCHMARK_NAME}.onnx" "{ROOT_DIR}/{BENCHMARK_NAME}.onnx"'
    # os.system(cmd)
    
    with torch.no_grad():
        dummy = torch.randn(10, 3, 32, 32).to(device)
        y1 = model(dummy)
        y2 = torch.from_numpy(inference_onnx(f'{ROOT_DIR}/{BENCHMARK_NAME}.onnx', dummy.detach().cpu().numpy())[0]).to(device)
        diff = torch.norm((y1 - y2).mean(0))
        print(f'{diff=}')
        assert torch.allclose(y1, y2, 1e-5, 1e-5)
        