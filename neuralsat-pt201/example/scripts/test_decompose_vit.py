import onnxruntime as ort
import onnx2pytorch
import numpy as np
import warnings
import logging
import torch
import tqdm
import onnx
import io

from torch.jit import ONNXTracedModule

from util.network.read_onnx import parse_onnx, decompose_onnx, decompose_pytorch


def split_network(net, input_shape, min_layer: int, device: str):
    split_idx = min_layer
    while True:
        prefix_onnx_byte, suffix_onnx_byte = decompose_pytorch(net, input_shape, split_idx + 1)
        net = net.to(device)
        if (prefix_onnx_byte is None) or (suffix_onnx_byte is None):
            return (net, input_shape, None), None
        assert prefix_onnx_byte is not None

        # move to next layer
        split_idx += 1
        
        # parse subnets
        prefix, prefix_input_shape, prefix_output_shape, _ = parse_onnx(prefix_onnx_byte)
        suffix, suffix_input_shape, suffix_output_shape, _ = parse_onnx(suffix_onnx_byte)

        # move to device
        prefix = prefix.to(device)
        suffix = suffix.to(device)
            
        # check correctness
        dummy = torch.randn(2, *prefix_input_shape[1:], device=device) # try batch=2
        if torch.allclose(net(dummy), suffix(prefix(dummy))):
            return (prefix.to(device), prefix_input_shape, prefix_output_shape), (suffix.to(device), suffix_input_shape, suffix_output_shape)
        else:
            print('Failed to decompose at layer:', split_idx)
            
            
if __name__ == "__main__":
    net_path = 'train/vit.onnx'
    net_path = 'train/vit-sim.onnx'
    net_path = 'train/vit_batch.onnx'
    # net_path = 'example/onnx/pgd_2_3_16.onnx'
    net = parse_onnx(net_path)[0]
    # print(net)
    # from train.models.vit import ViT_2_3_16
    x = torch.randn(1, 3, 32, 32)
    # net = ViT_2_3_16()
    # trace, _ = torch.jit._get_trace_graph(net, (x,))
    # trace_graph = trace.graph
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace, _ = torch.jit._get_trace_graph(net, (x,))
        # tracer = ONNXTracedModule(net)
        # print(tracer)
        print(trace)
        # tracer(x)
    
    # print(torch.jit.__file__)
