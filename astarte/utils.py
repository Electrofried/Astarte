# astarte/utils.py
import os
import time
import torch
from datetime import datetime

def detach_state(state):
    """
    Detaches each tensor in the state tuple to prevent gradient propagation.
    """
    return tuple(s.detach() for s in state)

def get_checkpoint_name(config):
    """
    Generate a checkpoint name with timestamp and key configuration parameters.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = f"L{config['num_layers']}_H{config['hidden_size']}_S{config['chunk_length']}"
    return f"checkpoint_{timestamp}_{params}"

def generate_dream(model, state, sequence_length=512, noise_std=0.01):
    """
    Generate a dream sequence from the current state.
    
    For each step:
        - Add Gaussian noise to the null channel.
        - Use the updated state to generate logits.
        - Sample a token from the probability distribution.
    
    Returns:
        List[int]: Generated token IDs.
    """
    dream_state = state
    dream_tokens = []
    for _ in range(sequence_length):
        # Add noise to the null channel (x0)
        noisy_null = dream_state[4] + torch.randn_like(dream_state[4]) * noise_std
        current_dream_state = (
            dream_state[0],
            dream_state[1],
            dream_state[2],
            dream_state[3],
            noisy_null
        )
        logits = model.generate_from_state(current_dream_state)
        probs = torch.softmax(logits, dim=-1)
        # Sample one token per batch; here we assume a batch size of 1 for simplicity
        token = torch.multinomial(probs, num_samples=1).item()
        dream_tokens.append(token)
    return dream_tokens
