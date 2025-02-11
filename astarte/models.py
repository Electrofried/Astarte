# astarte/models.py
import torch
import torch.nn as nn
from astarte.utils import detach_state

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

###############################################################################
# 1. Autonomic Base Pair Encoder (ABPE)
###############################################################################
class AutonomicBasePairEncoder(nn.Module):
    """
    Updates state registers using differential equations.
    
    Equations:
      x_A' = x_A + f_{u_A} * sin(ωt + φ) + λ * (p_A - (x_A - x_B))
      x_B' = x_B - f_{u_B} * sin(ωt + φ) + λ * (p_B - (x_B - x_A))
      p_A' = p_A + η * (x_B' - x_A')
      p_B' = p_B + η * (x_A' - x_B')
      x0'  = x0 + ζ * (|x_A' - x_B'| - x0)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.fu_A  = nn.Parameter(torch.tensor(0.1))
        self.fu_B  = nn.Parameter(torch.tensor(-0.2))
        self.lam   = nn.Parameter(torch.tensor(0.05))
        self.eta   = nn.Parameter(torch.tensor(0.01))
        self.zeta  = nn.Parameter(torch.tensor(0.1))
        self.omega = nn.Parameter(torch.tensor(1.0))
        self.phi   = nn.Parameter(torch.tensor(0.0))
        self.eps   = 1e-6

    def forward(self, x_A, x_B, p_A, p_B, x0, t):
        sin_val = torch.sin(self.omega * t + self.phi)
        new_x_A = x_A + self.fu_A * sin_val + self.lam * (p_A - (x_A - x_B))
        new_x_B = x_B - self.fu_B * sin_val + self.lam * (p_B - (x_B - x_A))
        new_p_A = p_A + self.eta * (new_x_B - new_x_A)
        new_p_B = p_B + self.eta * (new_x_A - new_x_B)
        diff = torch.abs(new_x_A - new_x_B)
        new_x0 = x0 + self.zeta * (diff - x0)
        return new_x_A, new_x_B, new_p_A, new_p_B, new_x0

###############################################################################
# 2. Autonomic Aggregated Attention Head (AAAH)
###############################################################################
def robust_median_pair(x_A, x_B):
    return (x_A + x_B) / 2.0

class AutonomicAggregatedAttentionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_ratio = nn.Parameter(torch.ones(4))
        self.state_mask  = nn.Parameter(torch.zeros(4))
        self.null_scale  = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_A, x_B, p_A, p_B, x0, group_mass, group_null):
        effective_mask = torch.softmax(self.state_mask, dim=-1)
        effective_weights = torch.softmax(self.state_ratio * effective_mask, dim=-1)
        mass = (effective_weights[0] * x_A +
                effective_weights[1] * x_B +
                effective_weights[2] * p_A +
                effective_weights[3] * p_B)
        active_signal = mass - group_mass
        diff_pair = x0 - mass
        group_diff = group_null - group_mass
        scaling_factor = torch.sigmoid(self.null_scale * (group_diff - diff_pair))
        final_attention = active_signal * scaling_factor
        return final_attention, mass

###############################################################################
# 3. Autonomic Attention Block (AAB)
###############################################################################
class AutonomicAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_attn_heads=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = AutonomicBasePairEncoder(hidden_size)
        self.attn_heads = nn.ModuleList([
            AutonomicAggregatedAttentionHead(hidden_size)
            for _ in range(num_attn_heads)
        ])
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * num_attn_heads, hidden_size).double(),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size).double()
        )
    
    def forward(self, state, t):
        new_state = self.encoder(*state, t)
        new_x_A, new_x_B, new_p_A, new_p_B, new_x0 = new_state
        
        # Compute group mass using the first attention head.
        attn_head0 = self.attn_heads[0]
        effective_mask = torch.softmax(attn_head0.state_mask, dim=-1)
        effective_weights = torch.softmax(attn_head0.state_ratio * effective_mask, dim=-1)
        group_mass = (effective_weights[0] * new_x_A +
                      effective_weights[1] * new_x_B +
                      effective_weights[2] * new_p_A +
                      effective_weights[3] * new_p_B)
        group_null = new_x0
        head_outputs = []
        for head in self.attn_heads:
            head_out, _ = head(new_x_A, new_x_B, new_p_A, new_p_B, new_x0,
                               group_mass, group_null)
            head_outputs.append(head_out)
        combined = torch.cat(head_outputs, dim=-1)
        attn_output = self.ffn(combined)
        return new_state, attn_output

###############################################################################
# 4. Autonomic Layer Stack (ALS)
###############################################################################
class AutonomicLayerStack(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attn_heads=1):
        super().__init__()
        self.layers = nn.ModuleList([
            AutonomicAttentionBlock(hidden_size, num_attn_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, initial_state, t_start=1.0, dt=1.0):
        state = initial_state
        for layer in self.layers:
            state, _ = layer(state, t_start)
            t_start += dt
        return state, None

###############################################################################
# 5. Autonomic Internal Vector Prediction Layer (AIVPL)
###############################################################################
class AutonomicInternalVectorPredictionLayer(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size * 3, vocab_size).double()
    
    def forward(self, state):
        x_A, x_B, p_A, p_B, x0 = state
        M_left  = robust_median_pair(x_A, p_A)
        M_right = robust_median_pair(x_B, p_B)
        d_left  = torch.norm(M_left - x0, dim=-1, keepdim=True)
        d_right = torch.norm(M_right - x0, dim=-1, keepdim=True)
        mask = (d_left < d_right).float()
        M_low  = mask * M_left + (1 - mask) * M_right
        M_high = mask * M_right + (1 - mask) * M_left
        V1 = M_low - x0
        V2 = M_high - x0
        V3 = M_high - M_low
        combined = torch.cat([V1, V2, V3], dim=-1)
        logits = self.proj(combined)
        return logits

###############################################################################
# 6. Autonomic Token Prediction Model (ATPM)
###############################################################################
class AutonomicTokenPredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_attn_heads):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size).double()
        self.input_proj = nn.Linear(embed_size, hidden_size).double()
        self.layer_stack = AutonomicLayerStack(hidden_size, num_layers, num_attn_heads)
        self.prediction_layer = AutonomicInternalVectorPredictionLayer(hidden_size, vocab_size)
        self.state_matrix = nn.Parameter(torch.eye(5).double())

    def process_state_as_matrix(self, state):
        state_mat = torch.stack(state, dim=1)
        processed = torch.einsum("ij,bjk->bik", self.state_matrix, state_mat)
        new_state = tuple(torch.unbind(processed, dim=1))
        return new_state

    def forward(self, input_ids, prev_state=None, pause=False, t_start=1.0, dt=1.0):
        if not pause:
            if prev_state is None:
                embedded = self.embedding(input_ids)
                obs = embedded.mean(dim=1)
                hidden_obs = self.input_proj(obs)
                x_A = hidden_obs
                x_B = torch.zeros_like(hidden_obs)
                p_A = torch.zeros_like(hidden_obs)
                p_B = torch.zeros_like(hidden_obs)
                x0  = torch.zeros_like(hidden_obs)
                state = (x_A, x_B, p_A, p_B, x0)
            else:
                embedded = self.embedding(input_ids)
                obs = embedded.mean(dim=1)
                hidden_obs = self.input_proj(obs)
                x_A = prev_state[0] + hidden_obs
                x_B, p_A, p_B, x0 = prev_state[1], prev_state[2], prev_state[3], prev_state[4]
                state = (x_A, x_B, p_A, p_B, x0)
            final_state, _ = self.layer_stack(state, t_start, dt)
            processed_state = self.process_state_as_matrix(final_state)
            logits = self.prediction_layer(processed_state)
            return logits, processed_state
        else:
            if prev_state is None:
                batch_size = input_ids.size(0)
                hidden_obs = self.input_proj(torch.zeros(batch_size, self.embed_size, device=input_ids.device))
                x_A = hidden_obs
                x_B = torch.zeros_like(hidden_obs)
                p_A = torch.zeros_like(hidden_obs)
                p_B = torch.zeros_like(hidden_obs)
                x0  = torch.zeros_like(hidden_obs)
                state = (x_A, x_B, p_A, p_B, x0)
            else:
                state = prev_state
            final_state, _ = self.layer_stack(state, t_start, dt)
            processed_state = self.process_state_as_matrix(final_state)
            logits = torch.zeros(input_ids.size(0), self.vocab_size, device=input_ids.device).double()
            return logits, processed_state

    def generate_from_state(self, state):
        processed_state = self.process_state_as_matrix(state)
        logits = self.prediction_layer(processed_state)
        return logits

    def null_cycle(self, prev_state, t_start=1.0, dt=1.0, alpha=0.5, beta=0.9, gamma=0.1):
        """
        Perform a retroactive null injection (roll-back) cycle using ratio-based deviation.
        
        1. For each channel s in {x_A, x_B, p_A, p_B}:
             r_s = s / (x0 + ε)
        2. Compute average deviation:
             Δ = (1/4) Σ |r_s - 1|
        3. Feedforward null update:
             x0_{t+1} = x0 + ζ (Δ - x0)
           (ζ is taken from the first layer's encoder)
        4. EMA smoothing:
             𝛥̄ = β 𝛥̄₍ₜ₋₁₎ + (1-β) Δ
        5. Retroactive null update:
             ẋ0 = (1-α)x0 + α x0_{t+1} + γ 𝛥̄
             
        New state S_new = (x_A_{t+1}, x_B_{t+1}, p_A_{t+1}, p_B_{t+1}, ẋ0)
        """
        # Feedforward update via layer stack to get S_{t+1}
        feed_state, _ = self.layer_stack(prev_state, t_start, dt)
        
        eps = 1e-8
        r_xA = prev_state[0] / (prev_state[4] + eps)
        r_xB = prev_state[1] / (prev_state[4] + eps)
        r_pA = prev_state[2] / (prev_state[4] + eps)
        r_pB = prev_state[3] / (prev_state[4] + eps)
        Delta = (torch.abs(r_xA - 1) + torch.abs(r_xB - 1) + torch.abs(r_pA - 1) + torch.abs(r_pB - 1)) / 4.0
        
        # Use zeta from the first layer's encoder
        zeta = self.layer_stack.layers[0].encoder.zeta
        new_x0_feed = prev_state[4] + zeta * (Delta - prev_state[4])
        
        # EMA smoothing of Delta
        if hasattr(self, "smoothed_delta"):
            smoothed_delta = beta * self.smoothed_delta + (1 - beta) * Delta
        else:
            smoothed_delta = Delta
        self.smoothed_delta = smoothed_delta
        
        # Retroactive null update
        new_x0 = (1 - alpha) * prev_state[4] + alpha * new_x0_feed + gamma * smoothed_delta
        
        new_state = (feed_state[0], feed_state[1], feed_state[2], feed_state[3], new_x0)
        processed_new_state = self.process_state_as_matrix(new_state)
        return processed_new_state, new_state
