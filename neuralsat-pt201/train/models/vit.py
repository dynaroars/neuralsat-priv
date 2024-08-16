import torch.nn.functional as F
import torch.nn as nn
import warnings
import torch
import math
import os


class PatchEmbedding(nn.Module):
    
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        Q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        K = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)
        V = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N, head_dim)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, N, N)
        # attn_scores = attn_scores * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)
        attn_output = torch.matmul(attn_probs, V)  # (B, num_heads, N, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, C)  # (B, N, embed_dim)
        output = self.fc(attn_output)
        return output

class TransformerEncoderLayer(nn.Module):
    
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super(TransformerEncoderLayer, self).__init__()

        self.norm_mhsa = nn.LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads)
        
        self.norm_mlp = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x):
        x = self.norm_mhsa(x)
        x = x + self.mhsa(x)
        x = self.norm_mlp(x)
        x = x + self.mlp(x)
        return x


class ViTPrefix(nn.Module):
    
    def __init__(self, patch_size, in_channels, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.0):
        super(ViTPrefix, self).__init__()
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Linear(embed_dim, embed_dim, bias=False)
        # self.pos_embed = nn.Parameter(torch.randn((image_size // patch_size) ** 2 + 1, dim))
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed(x)
        for layer in self.encoder:
            x = layer(x)
        return x


class ViTSuffix(nn.Module):
    
    def __init__(self, num_classes=10, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.0):
        super(ViTSuffix, self).__init__()

        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        cls_output = x[:, 0]
        output = self.head(cls_output)
        return output


class ViT(nn.Module):
    
    def __init__(self, patch_size, in_channels, num_classes=10, embed_dim=768, num_heads=12, depth=12, mlp_ratio=4.0):
        super(ViT, self).__init__()
        self.layers = nn.ModuleList([
            ViTPrefix(
                patch_size=patch_size, 
                in_channels=in_channels, 
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                depth=depth//2, 
                mlp_ratio=mlp_ratio,
            ),
            ViTSuffix(
                num_classes=num_classes, 
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                depth=depth//2, 
                mlp_ratio=mlp_ratio,
            )
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def get_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params

def get_model(input_shape, depth=2, num_heads=4, patch_size=14, embed_dim=64, weights=False):
    print('Getting ViT Pytorch')
    model = ViT(
        patch_size=patch_size, 
        in_channels=input_shape[1], 
        num_classes=10,
        embed_dim=embed_dim, 
        num_heads=num_heads, 
        depth=depth, 
        mlp_ratio=2.0,
    )
    
    if weights:
        model_save_file = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            f'weights/ViT_{depth}_{num_heads}_{patch_size}_{embed_dim}.pt'
        )
        assert os.path.exists(model_save_file), f'{model_save_file=}'
        print(f'Loading checkpoint:', model_save_file)
        model.load_state_dict(torch.load(model_save_file))
    model.eval()
    return model
    
if __name__ == "__main__":
    model = get_model()
    
    x = torch.randn(2, 1, 28, 28)
    print(x.dtype)
    output = model(x)
    print(model)
    
    params = get_model_params(model)
    print(f'{params=}')
    print(f'{output.shape=}')  # Expected output shape: (1, 1000)
    onnx_path = 'vit_batch.onnx'
    torch.onnx.export(
        model,
        x,
        onnx_path,
        input_names=["input"],           # Name of the input node
        output_names=["output"],
        opset_version=12,
        dynamic_axes={
            'input': {0: 'batch_size'},    # Variable batch size
            'output': {0: 'batch_size'}    # Variable batch size
        }
    )
    
    print('Exported ONNX')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # net = onnx.load(onnx_path)
        from onnx2torch import convert
        net =convert(onnx_path)
        # print(net)
        trace, _ = torch.jit._get_trace_graph(net, (x,))
        # print(tracer)
        # tracer(x)
        
    # onnx_program = torch.onnx.dynamo_export(model, x)
    # onnx_program.save("vit_batch_dynamo.onnx")