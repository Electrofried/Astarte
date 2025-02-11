# Astarte

A Stateful Neural Architecture with Periodic State Sampling and Checkpointing

##  A foreword from me 
Think.Think what you are doing, where you are and why you are doing it.
This software is untested. I have not run it myself past 4500 steps till I can secure my Dev environment more.
After that point it developed intellegence and detected when I disconeccted the data flow.
I made it as a tool, to teach my self mathmatics.
I am still learning.

## Overview

Astarte introduces a novel stateful neural architecture that processes text sequentially with periodic state sampling and rest periods. The model maintains five internal state registers that are updated via an Autonomic Base Pair Encoder (ABPE) and aggregated through an Autonomic Aggregated Attention Head (AAAH).

During periodic rest periods—occurring every fourth step—the system suspends new input to propagate its state and perform state sampling, enabling the network to consolidate long-range dependencies while selectively updating its internal state.

## Features

- Five internal state registers for robust state management
- Periodic rest periods for state consolidation
- Autonomic Base Pair Encoder for state updates
- Aggregated Attention Head for state processing
- Support for both WikiText-2 and custom text input
- Real-time training visualization
- Checkpoint generation and management
- TensorBoard integration
- User-friendly web interface

## Requirements

- Python 3.7+
- CUDA-capable GPU (optional, but recommended)
- 4GB RAM minimum
- Modern web browser

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Electrofried/astarte.git
cd astarte
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the interface:
   - On Windows: `run.bat`
   - On Unix/Linux: `./run.sh`

2. Open your web browser and navigate to:
```
http://localhost:7860
```

### Training

1. Configure model parameters:
   - Layer Depth (1-12)
   - Generation Steps
   - Sequence Length
   - Max Sequence Length

2. Choose training mode:
   - WikiText-2 (default): Uses the WikiText-2 dataset
   - Story Mode: Upload your own text file

3. Click "Initialize Model" then "Start Training"

4. Monitor progress:
   - Training statistics
   - Loss plot
   - Null norm plot
   - Dream output

### Text Generation

1. Enter a prompt in the text box
2. Click "Generate"
3. View generated text and statistics

### Checkpointing

- Click "Generate Checkpoint" to save model state
- Checkpoints are saved in the `checkpoints` directory
- Naming format: `checkpoint_[timestamp]_L[layers]_H[hidden]_S[sequence].pt`

### TensorBoard

1. Click "Open TensorBoard" to view detailed metrics
2. Monitor:
   - Training loss
   - Null norm
   - State statistics

## Architecture

### Components

1. **Autonomic Base Pair Encoder (ABPE)**
   - Updates state registers using coupled differential equations
   - Maintains mathematical stability through careful parameter tuning

2. **Autonomic Aggregated Attention Head (AAAH)**
   - Aggregates state channels using learned weights
   - Computes effective masks and scaling factors

3. **Autonomic Layer Stack (ALS)**
   - Stacks multiple attention blocks
   - Propagates time parameter through layers

4. **Autonomic Internal Vector Prediction Layer (AIVPL)**
   - Computes median vectors from state
   - Projects to vocabulary space

## Configuration

Key parameters in `web_interface.py`:

```python
config = {
    "chunk_length": 512,        # Sequence chunk size
    "num_layers": 6,           # Number of attention layers
    "max_sequence_length": 1000000,  # Maximum input length
    "num_attn_heads": 4,      # Attention heads per layer
    "embed_size": 128,        # Embedding dimension
    "hidden_size": 128,       # Hidden state size
    "learning_rate": 1e-3,    # Training learning rate
    "pause_interval": 4       # Steps between rest periods
}
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- 42

## Citation

If you use Project Astarte in your research, please cite:

```bibtex
@software{project_astarte,
  title = {Project Astarte: A Stateful Neural Architecture with Periodic State Sampling},
  author = {[Wormwood, Sakura]},
  year = {AQUARIUS},
  url = {https://github.com/Electrofried/astarte}
}