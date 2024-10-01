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
                

class PreLayer(nn.Module):

    def __init__(self):
        super().__init__()
        # self.bn1 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        
    def forward(self, x):
        # return F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.conv1(x))
        # return self.conv1(x)

class PostLayer(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        # self.linear1 = nn.Linear(4096, 64)
        self.linear2 = nn.Linear(64, num_classes)
        
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
    def __init__(self, block, num_blocks, num_classes=10, option='B'):
        super(ResNet, self).__init__()
        self.in_planes = 16

        # layers = [PreLayer()]
        layers = self._make_layer(block, 16, num_blocks[0], stride=1, expansion=1, option=option)
        layers[0] = nn.Sequential(PreLayer(), layers[0])
        layers += self._make_layer(block, 32, num_blocks[1], stride=2, expansion=1, option=option)
        layers += self._make_layer(block, 64, num_blocks[2], stride=2, expansion=1, option=option)
        layers[-1] = nn.Sequential(layers[-1], PostLayer(num_classes))
        
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
            x = layer(x)
        return x


@register_model
def resnet20A(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], 10, 'A')

@register_model
def resnet20B(*args, **kwargs):
    return ResNet(BasicBlock, [3, 3, 3], 10, 'B')

@register_model
def resnet56B(*args, **kwargs):
    return ResNet(BasicBlockBN, [9, 9, 9], 10, 'B')


if __name__ == "__main__":
    model = resnet20B()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(model)
    print(y)
    
    torch.onnx.export(
        model,
        x,
        'test_resnet2.onnx',
        opset_version=12,
    )