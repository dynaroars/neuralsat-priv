import torch.nn as nn
import torch

try:
    from timm.models.registry import register_model
except ModuleNotFoundError:
    def register_model(func):
        """
        Fallback wrapper in case timm isn't installed
        """
        return func
    
class ConvDeep(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
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
            nn.Flatten(1),
        )
        
        dim = self.feature(torch.randn(1, 3, 32, 32)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    

class ConvBase(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
        )
        
        dim = self.feature(torch.randn(1, 3, 32, 32)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    
    
class ConvWide(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
        )
        
        dim = self.feature(torch.randn(1, 3, 32, 32)).size(-1)
        
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
    
def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


@register_model
def conv_base(*args, **kwargs):
    return ConvBase()


@register_model
def conv_deep(*args, **kwargs):
    return ConvDeep()


@register_model
def conv_wide(*args, **kwargs):
    return ConvWide()



if __name__ == "__main__":
    
    device = 'cuda'
    model = conv_base().to(device)
    
    x = torch.randn(1, 3, 32, 32).to(device)
    print(model)
    print('#params:', get_model_params(model))
    y = model(x)
    print(y)
    