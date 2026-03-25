import torch
import torch.nn as nn
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

        # Only flux and passband are embedded
        self.flux_proj = nn.Linear(1, d_model)
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
        
        # --- MASKED GLOBAL AVERAGE POOLING ---
        if pad_mask is not None:
            # m_i \in {0, 1} where 1 is real data, 0 is padding
            m = (~pad_mask.squeeze(1).squeeze(1)).float()
            
            # Eq: Z = \sum (H_{L,i} * m_i) / \sum m_i
            valid_sum = torch.sum(H_L * m.unsqueeze(-1), dim=1)
            valid_count = torch.clamp(torch.sum(m, dim=1, keepdim=True), min=1.0)
            
            Z = valid_sum / valid_count
        else:
            Z = H_L.mean(dim=1)
        
        # Classification
        logits = self.classifier(Z)
        return logits
