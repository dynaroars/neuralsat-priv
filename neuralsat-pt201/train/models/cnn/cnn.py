import torch.nn as nn
import torch

class CNNBase(nn.Module):

    def __init__(self, num_classes):
        super(CNNBase, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Flatten(1),
            # nn.Dropout(0.5),
            nn.Linear(2304, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)
    
    
def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


if __name__ == "__main__":
    
    model = CNNBase(10)
    x = torch.randn(1, 3, 32, 32)
    print(model)
    print('#params:', get_model_params(model))
    y = model(x)
    print(y)
    