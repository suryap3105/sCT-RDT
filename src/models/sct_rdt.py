import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import sCTRDT_EncoderBlock

class Full_sCTRDT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model   = config['d_model']
        num_heads = config['num_heads']
        # FAILSAFE: validate divisibility before building the model so the error
        # is raised immediately at construction time with a clear message.
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        # Deep Non-Linear Flux Embedding
        self.flux_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # num_embeddings=7 covers passbands 0-5 (PLAsTiCC 6-band) plus an
        # overflow bucket (index 6) used by the clamp guard in forward().
        self._num_passbands = 6
        self.passband_emb = nn.Embedding(num_embeddings=self._num_passbands + 1, embedding_dim=d_model)

        # ABLATION FLAGS
        self.use_passband_emb = config.get('use_passband_emb', True)
        self.use_time_emb = config.get('use_time_emb', True)
        self.use_err_emb = config.get('use_err_emb', True)

        self.layers = nn.ModuleList([
            sCTRDT_EncoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                dropout=config['dropout']
            ) for _ in range(config['num_layers'])
        ])

        self.final_norm = nn.LayerNorm(d_model)
        
        # Attention Pooling Layer
        self.pooling_attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1)
        )
        
        self.classifier = nn.Linear(d_model, config['num_classes'])

    def forward(self, flux, passband, t_raw, err, pad_mask=None):
        # FAILSAFE: clamp passband IDs to [0, _num_passbands] so out-of-vocabulary
        # values (e.g. Kepler single-band encoded as 7) hit the overflow bucket
        # at index _num_passbands rather than causing an IndexError.
        passband = passband.clamp(0, self._num_passbands)
        
        # --- ABLATION OVERRIDES ---
        if not self.use_time_emb:
            t_raw = torch.zeros_like(t_raw)
        if not self.use_err_emb:
            err = torch.ones_like(err)
            
        # Input Projection
        H = self.flux_proj(flux.unsqueeze(-1))
        if self.use_passband_emb:
            H = H + self.passband_emb(passband)
        
        # Pass through N layers, routing time and error deep into the blocks
        for layer in self.layers:
            H = layer(H, t_raw, err, pad_mask)
            
        H_L = self.final_norm(H)
        
        # --- ATTENTION POOLING ---
        # Get scalar attention score for each timestep: [B, S, 1] -> [B, S]
        attn_scores = self.pooling_attention(H_L).squeeze(-1)
        
        if pad_mask is not None:
            # Mask out padding tokens (set score to -inf before softmax)
            m_pad = pad_mask.squeeze(1).squeeze(1) # True for padding
            attn_scores = attn_scores.masked_fill(m_pad, float('-inf'))
            
        # Convert scores to probabilities
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Failsafe: if an entire sequence is masked (all -inf), softmax produces NaNs
        # Replace NaNs with 0 to prevent crashing the batch
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Weighted sum: [B, S] x [B, S, d_model] -> [B, d_model]
        Z = torch.sum(H_L * attn_weights.unsqueeze(-1), dim=1)
        
        # Classification
        logits = self.classifier(Z)
        return logits
