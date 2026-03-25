import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaledContinuousRoPE(nn.Module):
    def __init__(self, d_k, base=10000.0, L_max=1000.0):
        super().__init__()
        self.L_max = L_max
        
        # Eq: \theta_c = b^{-2(c-1)/d_k}
        # Create frequencies for the d_k / 2 subspaces
        power_term = torch.arange(0, d_k, 2).float() / d_k
        self.register_buffer("inv_freq", 1.0 / (base ** power_term))

    def forward(self, q, k, t_raw):
        # Step 1: Input Normalization 
        # Eq: \tau_i = (t_i - min) / (max - min + \delta)
        t_min = t_raw.min(dim=-1, keepdim=True).values
        t_max = t_raw.max(dim=-1, keepdim=True).values
        t_norm = (t_raw - t_min) / (t_max - t_min + 1e-6)
        
        # Eq: \tilde{\tau}_i = \tau_i * L_max
        t_scaled = t_norm * self.L_max 
        
        # Calculate angles: \theta_c * \tilde{\tau}_i
        angles = torch.einsum("bs,d->bsd", t_scaled.to(self.inv_freq.device), self.inv_freq.to(t_scaled.device))
        
        # Duplicate angles for complex pairs (e.g., [a, b] -> [a, a, b, b])
        angles = torch.repeat_interleave(angles, 2, dim=-1) 
        angles = angles.unsqueeze(2) # Shape: [B, S, 1, d_k]
        
        cos_t = torch.cos(angles)
        sin_t = torch.sin(angles)
        
        # Helper function for rotation logic: (-x_2, x_1)
        def rotate_half(x):
            return torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).reshape_as(x)
            
        # Eq: \hat{q}_i = R_{\Theta}(\tilde{\tau}_i) q_i
        q_rotated = (q * cos_t) + (rotate_half(q) * sin_t)
        k_rotated = (k * cos_t) + (rotate_half(k) * sin_t)
        
        return q_rotated, k_rotated, t_scaled


class sCTRDT_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_out = nn.Linear(d_model, d_model)
        
        self.scRoPE = ScaledContinuousRoPE(self.d_k)
        
        # LTDK Parameter: \lambda_h
        self.lambda_h = nn.Parameter(torch.zeros(num_heads))
        
        # PEG Parameters: W_g and b_g
        self.W_g = nn.Linear(in_features=2, out_features=num_heads)
        self.gamma = 1e-9

    def forward(self, x, t_raw, error, pad_mask=None):
        B, S, _ = x.shape
        # Project inputs
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k)
        
        # --- K_periodic (scRoPE) ---
        q_rot, k_rot, t_scaled = self.scRoPE(q, k, t_raw)
        
        q_rot = q_rot.transpose(1, 2) # [B, num_heads, S, d_k]
        k_rot = k_rot.transpose(1, 2)
        v = v.transpose(1, 2)

        # Eq: (\hat{q}_i^T \hat{k}_j) / \sqrt{d_k}
        S_base = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # --- K_decay (LTDK) ---
        # Calculate |\tilde{\tau}_i - \tilde{\tau}_j| matrix
        time_gaps = torch.abs(t_scaled.unsqueeze(2) - t_scaled.unsqueeze(1)).unsqueeze(1)
        
        # Eq: \hat{\lambda}_h = \log(1 + \exp(\lambda_h))
        lambda_hat = F.softplus(self.lambda_h).view(1, self.num_heads, 1, 1)
        
        # Eq: D_{ij} = -\hat{\lambda}_h |\tilde{\tau}_i - \tilde{\tau}_j|
        K_decay = -(lambda_hat * time_gaps)
        
        # --- K_noise (PEG) ---
        # Create pairwise error matrix [\epsilon_i \oplus \epsilon_j]
        err_i = error.unsqueeze(2).expand(-1, -1, S).unsqueeze(-1)
        err_j = error.unsqueeze(1).expand(-1, S, -1).unsqueeze(-1)
        error_pairs = torch.cat([err_i, err_j], dim=-1) # Shape: (B, S, S, 2)
        
        # Eq: G_{ij} = \log( \sigma(W_g [\epsilon_i \oplus \epsilon_j] + b_g) + \gamma )
        gate_logits = self.W_g(error_pairs).permute(0, 3, 1, 2)
        gate_probs = torch.sigmoid(gate_logits)
        K_noise = torch.log(gate_probs + self.gamma)
        
        # --- THE MASTER EQUATION ---
        # Eq: S_{ij} = K_{periodic} + K_{decay} + K_{noise}
        S_mat = S_base + K_decay + K_noise
        
        # Apply padding mask M_{ij} (Set padded token logits to -infinity)
        if pad_mask is not None:
             S_mat = S_mat.masked_fill(pad_mask, float('-inf'))
        
        # Eq: A_{ij} = Softmax(S_{ij})
        A = F.softmax(S_mat, dim=-1)
        # FAILSAFE: rows where every key is masked produce all-(-inf) logits which
        # softmax maps to NaN.  Replace those with 0 so masked-average-pooling
        # downstream returns a valid zero vector for fully-padded sequences.
        A = torch.nan_to_num(A, nan=0.0)

        # Eq: Head_h = A @ V
        out = torch.matmul(A, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_out(out)
