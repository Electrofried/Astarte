"""
Project Astarte - A Stateful Neural Architecture with Periodic State Sampling
Copyright (C) 2025 Project Astarte Contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

interface.py: Core implementation of Project Astarte's neural architecture
Contains the model components and training logic.

Components:
- AutonomicBasePairEncoder (ABPE): Updates state registers
- AutonomicAggregatedAttentionHead (AAAH): Aggregates state channels
- AutonomicAttentionBlock (AAB): Combines ABPE and AAAH
- AutonomicLayerStack (ALS): Stacks multiple AABs
- AutonomicInternalVectorPredictionLayer (AIVPL): Generates predictions
"""

import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter

# Set device: use CUDA if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# Helper: Detach a state (tuple of tensors) so that gradients aren't propagated.
###############################################################################
def detach_state(state):
    """
    Detaches each tensor in the state tuple to prevent gradient propagation.
    
    Args:
        state (tuple): Tuple of tensors representing the model state
        
    Returns:
        tuple: New tuple with detached tensors
    """
    return tuple(s.detach() for s in state)

###############################################################################
# 1. Autonomic Base Pair Encoder (ABPE) in FP64
###############################################################################
class AutonomicBasePairEncoder(nn.Module):
    """
    Updates the state registers using coupled differential equations.
    
    The ABPE updates five state registers (x_A, x_B, p_A, p_B, x0) using:
      x_A' = x_A + fu_A*sin(ωt+φ) + λ*(p_A - (x_A - x_B))
      x_B' = x_B - fu_B*sin(ωt+φ) + λ*(p_B - (x_B - x_A))
      p_A' = p_A + η*(x_B' - x_A')
      p_B' = p_B + η*(x_A' - x_B')
      x0'  = x0 + ζ*(|x_A' - x_B'| - x0)
    
    All parameters and computations are in FP64 for numerical stability.
    """
    def __init__(self, hidden_size):
        super(AutonomicBasePairEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.fu_A  = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.fu_B  = nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))
        self.lam   = nn.Parameter(torch.tensor(0.05, dtype=torch.float64))
        self.eta   = nn.Parameter(torch.tensor(0.01, dtype=torch.float64))
        self.zeta  = nn.Parameter(torch.tensor(0.1, dtype=torch.float64))
        self.omega = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.phi   = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.eps   = 1e-6

    def forward(self, x_A, x_B, p_A, p_B, x0, t):
        """
        Forward pass of the ABPE.
        
        Args:
            x_A, x_B: Primary state registers
            p_A, p_B: Secondary state registers
            x0: Null channel register
            t: Time parameter
            
        Returns:
            tuple: Updated state registers
        """
        sin_val = torch.sin(self.omega * t + self.phi)
        new_x_A = x_A + self.fu_A * sin_val + self.lam * (p_A - (x_A - x_B))
        new_x_B = x_B - self.fu_B * sin_val + self.lam * (p_B - (x_B - x_A))
        new_p_A = p_A + self.eta * (new_x_B - new_x_A)
        new_p_B = p_B + self.eta * (new_x_A - new_x_B)
        diff = torch.abs(new_x_A - new_x_B)
        new_x0 = x0 + self.zeta * (diff - x0)
        return new_x_A, new_x_B, new_p_A, new_p_B, new_x0

###############################################################################
# 2. Autonomic Aggregated Attention Head (AAAH) in FP64
###############################################################################
def robust_median_pair(x_A, x_B):
    """Compute robust median of two vectors."""
    return (x_A + x_B) / 2.0

class AutonomicAggregatedAttentionHead(nn.Module):
    """
    Aggregates state channels using learned weights.
    
    The AAAH:
    1. Computes effective weights using softmax
    2. Aggregates state channels
    3. Computes active signal and differences
    4. Applies scaling based on group differences
    """
    def __init__(self, hidden_size):
        super(AutonomicAggregatedAttentionHead, self).__init__()
        self.hidden_size = hidden_size
        self.state_ratio = nn.Parameter(torch.ones(4, dtype=torch.float64))
        self.state_mask  = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        self.null_scale  = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, x_A, x_B, p_A, p_B, x0, group_mass, group_null):
        """
        Forward pass of the AAAH.
        
        Args:
            x_A, x_B, p_A, p_B: State registers
            x0: Null channel
            group_mass: Aggregated group mass
            group_null: Group null value
            
        Returns:
            tuple: (final_attention, mass)
        """
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
# 3. Autonomic Attention Block (AAB) in FP64
###############################################################################
class AutonomicAttentionBlock(nn.Module):
    """
    Combines ABPE and AAAH modules with a feed-forward network.
    
    The AAB:
    1. Updates state via ABPE
    2. Processes through multiple attention heads
    3. Combines outputs with feed-forward network
    """
    def __init__(self, hidden_size, num_attn_heads=1):
        super(AutonomicAttentionBlock, self).__init__()
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
        """
        Forward pass of the AAB.
        
        Args:
            state: Current model state
            t: Time parameter
            
        Returns:
            tuple: (new_state, attention_output)
        """
        new_state = self.encoder(*state, t)
        new_x_A, new_x_B, new_p_A, new_p_B, new_x0 = new_state

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
            head_out, mass = head(new_x_A, new_x_B, new_p_A, new_p_B, new_x0,
                                  group_mass, group_null)
            head_outputs.append(head_out)
        combined = torch.cat(head_outputs, dim=-1)
        attn_output = self.ffn(combined)
        return new_state, attn_output

###############################################################################
# 4. Autonomic Layer Stack (ALS) in FP64
###############################################################################
class AutonomicLayerStack(nn.Module):
    """
    Stacks multiple AABs sequentially.
    
    The ALS:
    1. Processes state through multiple attention blocks
    2. Updates time parameter between layers
    """
    def __init__(self, hidden_size, num_layers, num_attn_heads=1):
        super(AutonomicLayerStack, self).__init__()
        self.layers = nn.ModuleList([
            AutonomicAttentionBlock(hidden_size, num_attn_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, initial_state, t_start=1.0, dt=1.0):
        """
        Forward pass through the layer stack.
        
        Args:
            initial_state: Starting model state
            t_start: Initial time value
            dt: Time increment between layers
            
        Returns:
            tuple: (final_state, None)
        """
        state = initial_state
        for layer in self.layers:
            state, _ = layer(state, t_start)
            t_start += dt
        return state, None

###############################################################################
# 5. Autonomic Internal Vector Prediction Layer (AIVPL) in FP64
###############################################################################
class AutonomicInternalVectorPredictionLayer(nn.Module):
    """
    Generates predictions from internal state vectors.
    
    The AIVPL:
    1. Computes median vectors
    2. Identifies low/high masses
    3. Forms difference vectors
    4. Projects to vocabulary space
    """
    def __init__(self, hidden_size, vocab_size):
        super(AutonomicInternalVectorPredictionLayer, self).__init__()
        self.proj = nn.Linear(hidden_size * 3, vocab_size).double()
    
    def forward(self, state):
        """
        Forward pass of the AIVPL.
        
        Args:
            state: Current model state
            
        Returns:
            tensor: Vocabulary logits
        """
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
# 6. Autonomic Token Prediction Model (ATPM) with Rolling State in FP64
###############################################################################
class AutonomicTokenPredictionModel(nn.Module):
    """
    Main model class that processes input sequentially.
    
    The ATPM:
    1. Embeds input tokens
    2. Updates state based on input
    3. Processes through layer stack
    4. Generates predictions
    """
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_attn_heads):
        super(AutonomicTokenPredictionModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size).double()
        self.input_proj = nn.Linear(embed_size, hidden_size).double()
        self.layer_stack = AutonomicLayerStack(hidden_size, num_layers, num_attn_heads)
        self.prediction_layer = AutonomicInternalVectorPredictionLayer(hidden_size, vocab_size)
        self.state_matrix = nn.Parameter(torch.eye(5, dtype=torch.float64))

    def process_state_as_matrix(self, state):
        """Process state through learned 5x5 matrix."""
        state_mat = torch.stack(state, dim=1)
        processed = torch.einsum("ij,bjk->bik", self.state_matrix, state_mat)
        new_state = tuple(torch.unbind(processed, dim=1))
        return new_state

    def forward(self, input_ids, prev_state=None, pause=False, t_start=1.0, dt=1.0):
        """
        Forward pass of the ATPM.
        
        Args:
            input_ids: Input token IDs
            prev_state: Previous model state (if any)
            pause: Whether this is a rest period
            t_start: Initial time value
            dt: Time increment
            
        Returns:
            tuple: (logits, new_state)
        """
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
                x_B = prev_state[1]
                p_A = prev_state[2]
                p_B = prev_state[3]
                x0  = prev_state[4]
                state = (x_A, x_B, p_A, p_B, x0)
            final_state, _ = self.layer_stack(state, t_start, dt)
            processed_state = self.process_state_as_matrix(final_state)
            logits = self.prediction_layer(processed_state)
            return logits, processed_state
        else:
            if prev_state is None:
                batch_size = input_ids.size(0)
                hidden_obs = self.input_proj(torch.zeros(batch_size, self.embed_size,
                                                          device=input_ids.device,
                                                          dtype=torch.float64))
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
            logits = torch.zeros(input_ids.size(0), self.vocab_size,
                                 device=input_ids.device, dtype=torch.float64)
            return logits, processed_state

    def generate_from_state(self, state):
        """Generate prediction from current state."""
        processed_state = self.process_state_as_matrix(state)
        logits = self.prediction_layer(processed_state)
        return logits

###############################################################################
# 7. RollingTextDataset: Splitting the Story into Sequential Chunks
###############################################################################
class RollingTextDataset(Dataset):
    """
    Dataset class that handles text data with periodic rest periods.
    
    Every 2nd beat is a rest period:
    - Normal beat: returns (input_ids, target, pause=False)
    - Rest beat: returns dummy input with pause=True
    """
    def __init__(self, token_ids, chunk_length=512, pause_interval=2):
        self.token_ids = token_ids
        self.chunk_length = chunk_length
        self.pause_interval = pause_interval
        self.n_normal = len(token_ids) // chunk_length
        self.n_pause = self.n_normal // (pause_interval - 1)
        self.total_samples = self.n_normal + self.n_pause

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx % self.pause_interval == (self.pause_interval - 1):
            input_ids = torch.zeros(self.chunk_length, dtype=torch.long)
            target = -100
            pause = True
        else:
            k = idx - (idx // self.pause_interval)
            start = k * self.chunk_length
            end = start + self.chunk_length
            if end >= len(self.token_ids):
                chunk = self.token_ids[start:]
                padding = [0] * (self.chunk_length - len(chunk))
                input_ids = torch.tensor(chunk + padding, dtype=torch.long)
                target = 0
            else:
                chunk = self.token_ids[start:end]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                target = self.token_ids[end] if end < len(self.token_ids) else 0
            pause = False
        return input_ids, target, pause

###############################################################################
# 8. Dream Generation Function
###############################################################################
def generate_dream(model, state, sequence_length=512, noise_std=0.01):
    """
    Generate a dream sequence from the current state.
    
    Args:
        model: The ATPM model
        state: Current model state
        sequence_length: Length of sequence to generate
        noise_std: Standard deviation of noise to add
        
    Returns:
        list: Generated token IDs
    """
    dream_state = state
    dream_tokens = []
    for i in range(sequence_length):
        noisy_null = dream_state[4] + torch.randn_like(dream_state[4]) * noise_std
        current_dream_state = (dream_state[0],
                               dream_state[1],
                               dream_state[2],
                               dream_state[3],
                               noisy_null)
        logits = model.generate_from_state(current_dream_state)
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        dream_tokens.append(token)
    return dream_tokens
