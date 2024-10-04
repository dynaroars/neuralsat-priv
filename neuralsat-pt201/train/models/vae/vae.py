import torch
import torch.nn as nn
from .blocks import DownBlock, MidBlock, UpBlock


class VAE(nn.Module):
    def __init__(self, dataset_config, model_config):
        super().__init__()
        self.down_channels = model_config['down_channels']
        self.mid_channels = model_config['mid_channels']
        self.resample = model_config['resample']
        self.num_down_layers = model_config['num_down_layers']
        self.num_mid_layers = model_config['num_mid_layers']
        self.num_up_layers = model_config['num_up_layers']
        
        # To disable attention in Downblock of Encoder and Upblock of Decoder
        # Latent Dimension
        self.z_channels = model_config['z_channels']
        
        # Assertion to validate the channel information
        assert self.mid_channels[0] == self.down_channels[-1], f'{self.mid_channels[0]=} {self.down_channels[-1]=}'
        assert self.mid_channels[-1] == self.down_channels[-1]
        
        # Wherever we use downsampling in encoder correspondingly use
        # upsampling in decoder
        
        ##################### Encoder ######################
        self.encoder_conv_in = nn.Conv2d(dataset_config['im_channels'], self.down_channels[0], kernel_size=3, padding=(1, 1))
        
        # Downblock + Midblock
        self.encoder_layers = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.encoder_layers.append(DownBlock(in_channels=self.down_channels[i],
                                                 out_channels=self.down_channels[i + 1],
                                                 down_sample=self.resample,
                                                 num_layers=self.num_down_layers))
        
        self.encoder_mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.encoder_mids.append(MidBlock(in_channels=self.mid_channels[i], 
                                              out_channels=self.mid_channels[i + 1],
                                              num_layers=self.num_mid_layers))
        
        # self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        # self.encoder_norm_out = nn.Identity()
        # self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.z_channels, kernel_size=3, padding=1)
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], 2*self.z_channels, kernel_size=3, padding=1)

        # Latent Dimension is 2*Latent because we are predicting mean & variance
        # self.pre_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.pre_quant_conv = nn.Conv2d(2*self.z_channels, 2*self.z_channels, kernel_size=1)
        ####################################################
        
        
        ##################### Decoder ######################
        self.post_quant_conv = nn.Conv2d(self.z_channels, self.z_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.z_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        # Midblock + Upblock
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(in_channels=self.mid_channels[i], 
                                              out_channels=self.mid_channels[i - 1],
                                              num_layers=self.num_mid_layers))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(in_channels=self.down_channels[i], 
                                               out_channels=self.down_channels[i - 1],
                                               up_sample=self.resample,
                                               num_layers=self.num_up_layers))
        
        # self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[0])
        # self.decoder_norm_out = nn.Identity()
        self.decoder_conv_out = nn.Conv2d(self.down_channels[0], dataset_config['im_channels'], kernel_size=3, padding=1)
    
    def encode(self, x):
        out = self.encoder_conv_in(x)
        for down in self.encoder_layers:
            out = down(out)
        for mid in self.encoder_mids:
            out = mid(out)
        # out = self.encoder_norm_out(out)
        # out = nn.SiLU()(out)
        out = out.relu()
        out = self.encoder_conv_out(out)
        print(out.shape)
        out = self.pre_quant_conv(out)
        # return out
        mean, logvar = torch.chunk(out, 2, dim=1)
        # return mean, out
        std = torch.exp(0.5 * logvar)
        sample = mean + std * 0.1 # torch.randn_like(mean).to(x)
        return sample, out
    
    def decode(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        for mid in self.decoder_mids:
            out = mid(out)
        for up in self.decoder_layers:
            out = up(out)
        out = out.relu()
        out = self.decoder_conv_out(out)
        return out

    # def forward(self, x):
    #     z, encoder_output = self.encode(x)
    #     out = self.decode(z)
    #     return out, encoder_output


    def forward(self, x):
        z, _ = self.encode(x)
        out = self.decode(z)
        return out



if __name__ == "__main__":
    import yaml
    with open('config/cifar10.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    
    
    model = VAE(
        dataset_config=config['dataset_params'],
        model_config=config['autoencoder_params'],
    )
    
    print(model)
    
    x = torch.randn(1, 3, 32, 32)
    
    y = model(x)
    
    print(f'{x.shape=} {y.shape=}')
    
    torch.onnx.export(
        model,
        x,
        'vae_test.onnx',
        verbose=False,
        opset_version=12,
    )