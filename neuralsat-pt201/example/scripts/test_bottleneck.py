import torch.nn as nn
import torch

class Encoder(nn.Module):
    
    def __init__(self, in_channels, layer_configs, out_channels):
        super(Encoder, self).__init__()
        layers = []
        for out_channels, kernel_size, stride, padding in layer_configs + [(out_channels, 1, 1, 0)]:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            # layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    
    def __init__(self, in_channels, layer_configs, out_channels):
        super(Decoder, self).__init__()
        layers = []
        for out_channels, kernel_size, stride, padding in layer_configs + [(out_channels, out_channels, 1, 1)]:
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
            # layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    
    def __init__(self, in_channel, bottleneck_dim, encoder_configs):
        super(VAE, self).__init__()
        self.encoder = Encoder(in_channel, encoder_configs, bottleneck_dim)
        self.decoder = Decoder(bottleneck_dim, encoder_configs[::-1], in_channel)  # Adjust starting in_channels for decoder
        self.fc = nn.Linear(in_channel * 32 * 32, 10)  # Adjust according to your output size

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
        
    # Example usage
    encoder_layer_configs = [
        (4, 3, 1, 1),  # (out_channels, kernel_size, stride, padding)
        (6, 3, 1, 1),
        (8, 3, 1, 1),
    ]

    model = VAE(
        in_channel=3,
        bottleneck_dim=16,
        encoder_configs=encoder_layer_configs, 
    )
    print(model)

    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.shape)