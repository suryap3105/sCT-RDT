import torch.nn as nn
from .attention import sCTRDT_Attention

class sCTRDT_EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = sCTRDT_Attention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_raw, err, pad_mask=None):
        # Pre-LN Residual block for Attention
        x = x + self.dropout(self.attn(self.norm1(x), t_raw, err, pad_mask))
        # Pre-LN Residual block for FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x
