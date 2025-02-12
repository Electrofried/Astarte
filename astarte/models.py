# astarte/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

###############################################################################
# 1. Autonomic Base Pair Encoder (ABPE)
###############################################################################
class AutonomicBasePairEncoder(nn.Module):
    """
    Updates state registers using our helix‐driven differential equations.
    
    Equations:
      x_A' = x_A + f_{uA} * sin(ωt + φ) + λ * (p_A - (x_A - x_B))
      x_B' = x_B - f_{uB} * sin(ωt + φ) + λ * (p_B - (x_B - x_A))
      p_A' = p_A + η * (x_B' - x_A')
      p_B' = p_B + η * (x_A' - x_B')
      x₀'  = x₀ + ζ * (‖x_A' - x_B'‖ - x₀)
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
        # Compute the helix signal.
        s = torch.sin(self.omega * t + self.phi)
        # Update base registers.
        new_x_A = x_A + self.fu_A * s + self.lam * (p_A - (x_A - x_B))
        new_x_B = x_B - self.fu_B * s + self.lam * (p_B - (x_B - x_A))
        # Update momentum registers.
        new_p_A = p_A + self.eta * (new_x_B - new_x_A)
        new_p_B = p_B + self.eta * (new_x_A - new_x_B)
        # Update the null channel (drift update).
        diff = torch.norm(new_x_A - new_x_B, dim=-1, keepdim=True)
        new_x0 = x0 + self.zeta * (diff - x0)
        return new_x_A, new_x_B, new_p_A, new_p_B, new_x0

###############################################################################
# 2. Autonomic Aggregated Attention Head (AAAH)
###############################################################################
class AutonomicAggregatedAttentionHead(nn.Module):
    """
    Aggregates state registers and applies gravitational, lens, and self-scaling.
    
    Steps:
      A. Aggregation with self offset:
         m_raw = w_0*x_A' + w_1*x_B' + w_2*p_A' + w_3*p_B'
         group_mass_local = m_raw + x₀'
      
      B. Gravitational & Lens Scaling:
         L = α * sigmoid(τ * head_mass) + β
      
      C. Self-Scaling:
         S = sigmoid(γ * (group_mass_local - x₀') + δ)
      
      Final attention = (group_mass_local - external group_mass) * L * S
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Aggregation parameters.
        self.state_ratio = nn.Parameter(torch.ones(4))
        self.state_mask  = nn.Parameter(torch.zeros(4))
        # Learned parameters for gravitational dynamics and lens scaling.
        self.head_mass = nn.Parameter(torch.tensor(1.0))   # Gravitational mass.
        self.tau = nn.Parameter(torch.tensor(1.0))         # Lens thickness.
        self.gamma = nn.Parameter(torch.tensor(1.0))       # Self-scaling gamma.
        self.delta = nn.Parameter(torch.tensor(0.0))       # Self-scaling delta.
        self.alpha = nn.Parameter(torch.tensor(1.0))       # Lens scaling alpha.
        self.beta  = nn.Parameter(torch.tensor(0.1))       # Lens baseline beta.

    def forward(self, x_A, x_B, p_A, p_B, x0, group_mass, group_null):
        # A. Aggregation with learned weights.
        effective_mask = F.softmax(self.state_mask, dim=-1)
        effective_weights = F.softmax(self.state_ratio * effective_mask, dim=-1)
        mass_raw = (effective_weights[0] * x_A +
                    effective_weights[1] * x_B +
                    effective_weights[2] * p_A +
                    effective_weights[3] * p_B)
        # Add the null offset for self-awareness.
        group_mass_local = mass_raw + x0
        active_signal = group_mass_local - group_mass

        # B. Gravitational & Lens Scaling.
        G = self.head_mass
        L = self.alpha * torch.sigmoid(self.tau * G) + self.beta

        # C. Self-Scaling.
        S = torch.sigmoid(self.gamma * (group_mass_local - x0) + self.delta)
        
        final_attention = active_signal * L * S
        return final_attention, group_mass_local

###############################################################################
# 3. Autonomic Attention Block (AAB)
###############################################################################
class AutonomicAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_attn_heads=1, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = AutonomicBasePairEncoder(hidden_size)
        self.attn_heads = nn.ModuleList([
            AutonomicAggregatedAttentionHead(hidden_size)
            for _ in range(num_attn_heads)
        ])
        # Update LayerNorm to cover the concatenated dimension.
        self.layernorm = nn.LayerNorm(hidden_size * num_attn_heads)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size * num_attn_heads, hidden_size).double(),
            nn.ReLU(),
            self.dropout,
            nn.Linear(hidden_size, hidden_size).double()
        )
    
    def forward(self, state, t):
        new_state = self.encoder(*state, t)
        new_x_A, new_x_B, new_p_A, new_p_B, new_x0 = new_state
        
        # Compute a reference group mass using the first attention head.
        attn_head0 = self.attn_heads[0]
        effective_mask = F.softmax(attn_head0.state_mask, dim=-1)
        effective_weights = F.softmax(attn_head0.state_ratio * effective_mask, dim=-1)
        group_mass = (effective_weights[0] * new_x_A +
                      effective_weights[1] * new_x_B +
                      effective_weights[2] * new_p_A +
                      effective_weights[3] * new_p_B) + new_x0
        group_null = new_x0

        head_outputs = []
        for head in self.attn_heads:
            head_out, _ = head(new_x_A, new_x_B, new_p_A, new_p_B, new_x0, group_mass, group_null)
            head_outputs.append(head_out)
        combined = torch.cat(head_outputs, dim=-1)
        combined = self.layernorm(combined)
        attn_output = self.ffn(combined)
        return new_state, attn_output

###############################################################################
# 4. Autonomic Layer Stack (ALS)
###############################################################################
class AutonomicLayerStack(nn.Module):
    def __init__(self, hidden_size, num_layers, num_attn_heads=1, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            AutonomicAttentionBlock(hidden_size, num_attn_heads, dropout)
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
        # Projects from the high pair difference to vocabulary logits.
        self.proj = nn.Linear(hidden_size, vocab_size).double()
    
    def forward(self, state):
        x_A, x_B, p_A, p_B, x0 = state
        # Compute pair medians.
        M_A = (x_A + p_A) / 2.0
        M_B = (x_B + p_B) / 2.0
        # Measure distances from null.
        d_A = torch.norm(M_A - x0, dim=-1, keepdim=True)
        d_B = torch.norm(M_B - x0, dim=-1, keepdim=True)
        # Select the "high" pair (farthest from x0).
        high_mask = (d_A > d_B).float()
        M_high = high_mask * M_A + (1 - high_mask) * M_B
        # Form prediction vector and project.
        V = M_high - x0
        logits = self.proj(V)
        return logits

###############################################################################
# 6. Autonomic Token Prediction Model (ATPM)
###############################################################################
class AutonomicTokenPredictionModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_attn_heads, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size).double()
        self.input_proj = nn.Sequential(
            nn.Linear(embed_size, hidden_size).double(),
            nn.LayerNorm(hidden_size)
        )
        self.layer_stack = AutonomicLayerStack(hidden_size, num_layers, num_attn_heads, dropout)
        self.prediction_layer = AutonomicInternalVectorPredictionLayer(hidden_size, vocab_size)
        self.state_matrix = nn.Parameter(torch.eye(5).double())

    def process_state_as_matrix(self, state):
        state_mat = torch.stack(state, dim=1)
        processed = torch.einsum("ij,bjk->bik", self.state_matrix, state_mat)
        new_state = tuple(torch.unbind(processed, dim=1))
        return new_state

    def forward(self, input_ids, prev_state=None, t_start=1.0, dt=1.0):
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
            # Write new observation only to x_A.
            x_A = prev_state[0] + hidden_obs
            x_B, p_A, p_B, x0 = prev_state[1], prev_state[2], prev_state[3], prev_state[4]
            state = (x_A, x_B, p_A, p_B, x0)
        final_state, _ = self.layer_stack(state, t_start, dt)
        processed_state = self.process_state_as_matrix(final_state)
        logits = self.prediction_layer(processed_state)
        return logits, processed_state

    def generate_from_state(self, state):
        processed_state = self.process_state_as_matrix(state)
        logits = self.prediction_layer(processed_state)
        return logits
