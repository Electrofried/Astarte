# astarte/utils.py
import torch

def detach_state(state):
    """Detach each tensor in the state tuple."""
    return tuple(s.detach() for s in state)

def get_checkpoint_name(config):
    """Generate a checkpoint name using a timestamp and key config parameters."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = f"L{config['num_layers']}_H{config['hidden_size']}_S{config['chunk_length']}"
    return f"checkpoint_{timestamp}_{params}"

def generate_dream(model, state, sequence_length=512):
    """
    Generate an autoregressive sequence ("dream") from the current state.
    
    For each step, the function:
      1. Obtains logits via model.generate_from_state.
      2. Samples a token from the softmax distribution.
      3. Updates the state using the sampled token.
    """
    current_state = state
    generated_tokens = []
    for _ in range(sequence_length):
        logits = model.generate_from_state(current_state)
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        generated_tokens.append(token)
        dummy_input = torch.tensor([[token]], dtype=torch.long, device=logits.device)
        _, current_state = model(dummy_input, prev_state=current_state)
    return generated_tokens
