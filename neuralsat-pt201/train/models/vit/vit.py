import torch.nn as nn
import torch

from .vit_utils import Tokenizer, TransformerClassifier
from timm.models.registry import register_model
            
class ViTLite(nn.Module):
    def __init__(self,
                 img_size=224,
                 embedding_dim=768,
                 n_input_channels=3,
                 kernel_size=16,
                 dropout=0.,
                 attention_dropout=0.1,
                 stochastic_depth=0.1,
                 num_layers=14,
                 num_heads=6,
                 mlp_ratio=4.0,
                 num_classes=1000,
                 positional_embedding='learnable',
                 *args, **kwargs):
        super(ViTLite, self).__init__()
        assert img_size % kernel_size == 0, f"Image size ({img_size}) has to be" \
                                            f"divisible by patch size ({kernel_size})"
        self.tokenizer = Tokenizer(n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim,
                                   kernel_size=kernel_size,
                                   stride=kernel_size,
                                   padding=0,
                                   max_pool=False,
                                   activation=None,
                                   n_conv_layers=1,
                                   conv_bias=True)

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=n_input_channels, 
                                                           height=img_size, 
                                                           width=img_size),
            embedding_dim=embedding_dim,
            seq_pool=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=num_classes,
            positional_embedding=positional_embedding
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)

def _vit_lite(num_layers, num_heads, mlp_ratio, embedding_dim,
              positional_embedding='learnable',
              kernel_size=4, *args, **kwargs):
    model = ViTLite(num_layers=num_layers,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    embedding_dim=embedding_dim,
                    kernel_size=kernel_size,
                    positional_embedding=positional_embedding,
                    *args, **kwargs)

    return model


@register_model
def vit_2_4(img_size=32, positional_embedding='sine', num_classes=10, *args, **kwargs):
    return _vit_lite(
        num_layers=4, 
        num_heads=4,
        mlp_ratio=2, 
        embedding_dim=128, 
        kernel_size=8,
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )

@register_model
def vit_7_4_256(img_size=32, positional_embedding='sine', num_classes=10, *args, **kwargs):
    return _vit_lite(
        num_layers=7, 
        num_heads=4,
        mlp_ratio=2, 
        embedding_dim=256, 
        kernel_size=4,
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )

@register_model
def vit_6_4_128(img_size=32, positional_embedding='sine', num_classes=10, *args, **kwargs):
    return _vit_lite(
        num_layers=6, 
        num_heads=4,
        mlp_ratio=2, 
        embedding_dim=128, 
        kernel_size=4,
        img_size=img_size, 
        positional_embedding=positional_embedding,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
        *args, 
        **kwargs
    )
    
    
if __name__ == "__main__":
    vit = vit_2_4()
    
    print(vit)