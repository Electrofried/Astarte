# astarte/models.py
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)  # Ensure FP64 precision for numerical stability

###############################################################################
# 1. Autonomic Base Pair Encoder (ABPE)
###############################################################################
class AutonomicBasePairEncoder(nn.Module):
    """
    Updates state registers using coupled differential equations.

    Equations:
        x_A' = x_A + fu_A * sin(ωt+φ) + λ * (p_A - (x_A - x_B))
        x_B' = x_B - fu_B * sin(ωt+φ) + λ * (p_B - (x_B - x_A))
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
    """Compute robust median of two vectors."""
    return (x_A + x_B) / 2.0

class AutonomicAggregatedAttentionHead(nn.Module):
    """
    Aggregates state channels using learned weights.

    Math:
        effective_mask = softmax(state_mask)
        effective_weights = softmax(state_ratio * effective_mask)
        mass = sum(effective_weights[i] * state_i)  for i in {x_A, x_B, p_A, p_B}
        active_signal = mass - group_mass
        scaling_factor = sigmoid(null_scale * (group_null - group_mass - (x0 - mass)))
        final_attention = active_signal * scaling_factor
    """
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
    """
    Combines the ABPE and AAAH with a feed-forward network.
    """
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

        # Use the first attention head to compute a group mass
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
    """
    Stacks multiple Attention Blocks sequentially.
    """
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
    """
    Generates predictions from internal state vectors.

    Math:
        M_left  = median(x_A, p_A)
        M_right = median(x_B, p_B)
        Then, V1 = M_low - x0, V2 = M_high - x0, V3 = M_high - M_low,
        and the final logits are given by a linear projection on [V1,V2,V3].
    """
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
    """
    Main model class that sequentially processes input tokens.

    It embeds the input, updates its state through a layer stack, and generates
    vocabulary predictions via the AIVPL.
    """
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
        """Process state through a learned 5x5 matrix."""
        state_mat = torch.stack(state, dim=1)
        processed = torch.einsum("ij,bjk->bik", self.state_matrix, state_mat)
        new_state = tuple(torch.unbind(processed, dim=1))
        return new_state

    def forward(self, input_ids, prev_state=None, pause=False, t_start=1.0, dt=1.0):
        if not pause:
            if prev_state is None:
                # When no previous state is provided, initialize with the input.
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
                # Update only the first state channel with new observation
                x_A = prev_state[0] + hidden_obs
                x_B, p_A, p_B, x0 = prev_state[1], prev_state[2], prev_state[3], prev_state[4]
                state = (x_A, x_B, p_A, p_B, x0)
            final_state, _ = self.layer_stack(state, t_start, dt)
            processed_state = self.process_state_as_matrix(final_state)
            logits = self.prediction_layer(processed_state)
            return logits, processed_state
        else:
            # In pause mode, skip the usual prediction
            if prev_state is None:
                batch_size = input_ids.size(0)
                hidden_obs = self.input_proj(torch.zeros(batch_size, self.embed_size,
                                                          device=input_ids.device))
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
        """Generate predictions (logits) from the current state."""
        processed_state = self.process_state_as_matrix(state)
        logits = self.prediction_layer(processed_state)
        return logits

    def null_cycle(self, prev_state, aggregated_null, t_start=1.0, dt=1.0, alpha=0.5):
        """
        Perform the null injection cycle:
        - Write-Out Phase: Propagate the state through the layer stack to update x0:
            x0' = x0 + ζ(|x_A' - x_B'| - x0)
        - Write-In Phase: Re-inject the aggregated null:
            x0'' = x0' + α(aggregated_null - x0')
        
        Args:
            prev_state: Previous state tuple (x_A, x_B, p_A, p_B, x0)
            aggregated_null: The aggregated null value (tensor) computed externally.
            t_start: Initial time.
            dt: Time increment.
            alpha: Mixing parameter.
        
        Returns:
            processed_new_state: Processed state after updating the null channel.
            new_state: New state tuple with updated null channel.
        """
        # Write-Out: Propagate state through layer stack
        final_state, _ = self.layer_stack(prev_state, t_start, dt)
        processed_state = self.process_state_as_matrix(final_state)
        # Write-In: Update null channel
        x0_updated = final_state[4] + alpha * (aggregated_null - final_state[4])
        new_state = (final_state[0], final_state[1], final_state[2], final_state[3], x0_updated)
        processed_new_state = self.process_state_as_matrix(new_state)
        return processed_new_state, new_state
