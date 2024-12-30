import torch.nn.functional as F
import torch.nn as nn
import torch

from .resnet_utils import _weights_init, BasicBlock, BasicBlockBN

try:
    from timm.models.registry import register_model
except ModuleNotFoundError:
    def register_model(func):
        """
        Fallback wrapper in case timm isn't installed
        """
        return func
                

    
def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


class PreLayer(nn.Module):

    def __init__(self, planes):
        super().__init__()
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(3, planes, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        # return F.relu(self.bn1(self.conv1(x)))
        # return F.relu(self.conv1(x))
        return self.conv1(x)

class PostLayer(nn.Module):

    def __init__(self, dim, num_classes=10):
        super().__init__()
        # self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # x = x.relu()
        # print(x.shape)
        x = x.mean(dim=[2, 3]) # average pooling
        # print(x.shape)
        # x = x.view(x.shape[0], -1)
        # x = self.linear2(self.linear1(x).relu())
        x = self.linear2(x)
        return x
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, option='B', in_planes=16, hidden_planes=32):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.hidden_planes = hidden_planes
        pre_layer = PreLayer(self.in_planes)
        # layers = [PreLayer()]
        assert len(num_blocks)
        layers = self._make_layer(block, self.hidden_planes, num_blocks[0], stride=1, expansion=1, option=option)
        layers[0] = nn.Sequential(pre_layer, layers[0])
        if len(num_blocks) > 1:
            layers += self._make_layer(block, self.hidden_planes, num_blocks[1], stride=2, expansion=1, option=option)
        if len(num_blocks) > 2:
            layers += self._make_layer(block, self.hidden_planes, num_blocks[2], stride=2, expansion=1, option=option)
        layers[-1] = nn.Sequential(layers[-1], PostLayer(dim=self.hidden_planes, num_classes=num_classes))
        
        self.layers = nn.ModuleList(layers)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, option, expansion):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_planes=self.in_planes, 
                    planes=planes, 
                    stride=stride, 
                    option=option,
                    expansion=expansion,
                )
            )
            self.in_planes = planes * expansion
        return layers

    def forward(self, x):
        for layer in self.layers:
            # print(f'{x.shape=}')
            x = layer(x)
        return x



@register_model
def resnet_base(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[1, 1, 1],
        num_classes=10, 
        option='B', 
        in_planes=32, 
        hidden_planes=24)



@register_model
def resnet_deep(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[1, 2, 3],
        num_classes=10, 
        option='B', 
        in_planes=32, 
        hidden_planes=24)



@register_model
def resnet_deep_2(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[2, 2, 3],
        num_classes=10, 
        option='B', 
        in_planes=32, 
        hidden_planes=24)


@register_model
def resnet_deep_3(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[3, 3, 3],
        num_classes=10, 
        option='B', 
        in_planes=32, 
        hidden_planes=24)


@register_model
def resnet_wide(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[1, 1, 1],
        num_classes=10, 
        option='B', 
        in_planes=64, 
        hidden_planes=64)


@register_model
def resnet_wide_2(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[1, 1, 1],
        num_classes=10, 
        option='B', 
        in_planes=64, 
        hidden_planes=96)

@register_model
def resnet_toy(*args, **kwargs):
    return ResNet(
        block=BasicBlock, 
        num_blocks=[1, 1, 1],
        num_classes=10, 
        option='B', 
        in_planes=64, 
        hidden_planes=64)


if __name__ == "__main__":
    model = resnet_toy()
    x = torch.randn(1, 3, 32, 32)
    print(model)
    print('#params:', get_model_params(model))
    y = model(x)
    print(y)
    
    # torch.onnx.export(
    #     model,
    #     x,
    #     'test_resnet2.onnx',
    #     opset_version=12,
    # )