Please review:

---

# Project Astarte  
## A Stateful Neural Architecture with Periodic State Sampling

**Foreword:**  
*Think. Think what you are doing, where you are and why you are doing it.*  
I made this tool to teach myself mathematics. I am still learning.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Mathematical Foundations](#mathematical-foundations)
    - [Base Update Equations](#base-update-equations)
    - [Null Channel Write-Out Phase](#null-channel-write-out-phase)
    - [Null Injection (Write-In) Phase](#null-injection-write-in-phase)
    - [Normalization and Aggregation](#normalization-and-aggregation)
    - [Epoch Training with Fixed-Length Chunks](#epoch-training-with-fixed-length-chunks)
4. [Code Structure](#code-structure)
5. [Configuration Settings](#configuration-settings)
6. [DDNA Analogy](#ddna-analogy)
7. [Installation and Setup](#installation-and-setup)
8. [Usage Instructions](#usage-instructions)
    - [Running the Interface](#running-the-interface)
    - [Training the Model](#training-the-model)
    - [Generation and Checkpointing](#generation-and-checkpointing)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)

---

## 1. Introduction

Project Astarte introduces a new kind of stateful neural architecture that integrates periodic state sampling with a central “null channel.” This mechanism, updated in two phases (write‐out and write‐in), provides a robust method for preserving and evolving the internal state over multiple epochs. The design allows for precise checkpointing and fine-tuning, making it not only a powerful model but also an educational tool for understanding advanced mathematical concepts in neural network design.

---

## 2. Project Overview

Project Astarte processes sequential text data by breaking it into fixed-length chunks (epochs) and updating an internal state that consists of multiple channels. The key components are:

- **Base Update Module (ABPE):** Uses differential equations to update primary and secondary state channels.
- **Null Channel Mechanism:** A central channel, \( x_0 \), updated in two phases:
  - **Write-Out Phase:** Updates based on the difference between primary channels.
  - **Write-In Phase:** Re-injects an aggregated normalized null value.
- **Attention and Layer Stack:** Aggregates state information across layers.
- **Epoch Training:** Processes fixed-length token chunks repeatedly.
- **Checkpointing:** Saves model parameters and the null channel history for exact state reconstruction.
- **User Interface:** A Gradio-based UI for configuration, training, generation, and checkpointing.

---

## 3. Mathematical Foundations

### Base Update Equations

The model maintains five state registers:
- \( x_A \) and \( x_B \): Primary state channels.
- \( p_A \) and \( p_B \): Secondary (momentum) channels.
- \( x_0 \): The null channel (the central backbone).

The base update equations are:

$$
x_A' = x_A + f_{u_A} \sin(\omega t + \phi) + \lambda \left( p_A - \left( x_A - x_B \right) \right)
$$

$$
x_B' = x_B - f_{u_B} \sin(\omega t + \phi) + \lambda \left( p_B - \left( x_B - x_A \right) \right)
$$

$$
p_A' = p_A + \eta \left( x_B' - x_A' \right)
$$

$$
p_B' = p_B + \eta \left( x_A' - x_B' \right)
$$

*These equations are implemented in the `AutonomicBasePairEncoder.forward` method in* **`astarte/models.py`**.

---

### Null Channel Write-Out Phase

After updating the base channels, the null channel is updated using:

$$
x_0' = x_0 + \zeta \left( \left| x_A' - x_B' \right| - x_0 \right)
$$

*This update is part of the same forward pass in the base update module and is critical for capturing differences between the primary channels.*

---

### Null Injection (Write-In) Phase

In the null injection phase, the model re-injects an aggregated null value into the updated null channel. This is defined as:

$$
x_0'' = x_0' + \alpha \left( \bar{x}_0 - x_0' \right)
$$

*Here, \( \alpha \) (configured as `null_mix_alpha`) controls the degree to which the aggregated null value \( \bar{x}_0 \) overrides the current state \( x_0' \). This mechanism is implemented in the `null_cycle` method in* **`astarte/models.py`**.

---

### Normalization and Aggregation

The null outputs from each cycle \( x_0^{(i)} \) are normalized using min–max normalization:

$$
\text{Norm}(x_0^{(i)}) = \frac{x_0^{(i)} - \min(x_0)}{\max(x_0) - \min(x_0) + \epsilon}
$$

Then, the aggregated null value is computed as:

$$
\bar{x}_0 = \frac{1}{N} \sum_{i=1}^{N} \text{Norm}(x_0^{(i)})
$$

*This is handled by the `compute_aggregated_null` function in* **`astarte/web_interface.py`**.

---

### Epoch Training with Fixed-Length Chunks

Input text is divided into fixed-length chunks (or epochs):

$$
T = \{ t_1, t_2, \dots, t_L \}
$$

Each chunk is processed to update the state:

$$
\text{State}_{\text{new}} = f\Bigl( \text{State}_{\text{old}},\, T,\, x_0 \Bigr)
$$

The loss for each chunk is computed (e.g., using cross-entropy), and an epoch consists of multiple such chunks.

*The chunking mechanism is implemented in the `RollingTextDataset` class in* **`astarte/dataset.py`**.

---

## 4. Code Structure

The repository is structured as follows:

```
project_astarte/
├── astarte/
│   ├── __init__.py
│   ├── models.py              # Neural architecture (ABPE, AAAH, AAB, ALS, AIVPL, ATPM)
│   ├── dataset.py             # RollingTextDataset for fixed-length token chunks
│   ├── utils.py               # Helper functions (detach_state, generate_dream, get_checkpoint_name)
│   └── web_interface.py       # Training loop, null injection, state aggregation, and checkpointing
├── gradio_interface.py        # Gradio UI for configuration, training, and generation
└── README.md                  # This document
```

Each file corresponds to a component of the overall architecture, mapping directly to the mathematical formulations described above.

---

## 5. Configuration Settings

Key configuration parameters (in **`astarte/web_interface.py`**) include:

- **`chunk_length`**: Length of each token chunk.
- **`num_layers`**: Number of layers in the layer stack.
- **`max_sequence_length`**: Maximum allowed input sequence length.
- **`num_attn_heads`**: Number of attention heads per block.
- **`embed_size`**: Dimensionality of token embeddings.
- **`hidden_size`**: Size of the hidden state channels.
- **`learning_rate`**: Base learning rate for the optimizer.
- **`t_start`** & **`dt`**: Time parameters affecting the sinusoidal state updates.
- **`dream_noise_std`** & **`dream_sequence_length`**: Parameters for generating new tokens ("dreaming").
- **`generation_steps`**: Number of steps during generation.
- **`pause_interval`**: Interval between pause (rest) periods in training.
- **`checkpoint_dir`**: Directory for saving checkpoints.
- **`null_injection_token`**: Token used for initializing the null injection process.
- **`null_mix_alpha`**: The epoch learning rate \( \alpha \) used during the null injection phase. This parameter controls how strongly the aggregated null \( \bar{x}_0 \) is mixed into the current null state \( x_0' \).

These settings can be adjusted via the Gradio UI (see the Configuration tab in the interface).

---

## 6. DDNA Analogy

Project Astarte is inspired by the structure and function of DNA, especially the concept of DDNA (Dynamic DNA). Here’s how the analogy applies:

- **Helix Structure and Backbone:**  
  DNA’s double helix is supported by a central backbone (the sugar-phosphate backbone). In Astarte, the null channel \( x_0 \) serves as the invariant core or backbone of the network's internal state.

- **Null Channel as Genetic Code:**  
  The two-phase update of the null channel (write-out and write-in) mirrors how genetic information is maintained and expressed. The aggregated null \( \bar{x}_0 \) is analogous to a stable genetic code that influences the model's behavior.

- **Epoch Training and Repetitive Expression:**  
  Just as genes are expressed repeatedly over time, the model processes fixed-length token chunks (epochs) repeatedly. This repetitive expression reinforces and refines the internal state.

- **Configuration as Genetic Regulation:**  
  Configuration parameters (e.g., \( \text{null_mix_alpha} \)) regulate the influence of the null channel on the overall state. This is similar to how genetic regulators control gene expression.

This analogy helps clarify the innovative approach of Astarte and its potential to serve as a teaching tool for advanced mathematical and neural network concepts.

---

## 7. Installation and Setup

### Prerequisites

- **Python 3.8+**
- **PyTorch** (with CUDA support if desired; CPU-only mode works as well)
- **Other Dependencies:**  
  Transformers, Gradio, Datasets, Matplotlib, Chardet

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/project_astarte.git
   cd project_astarte
   ```

2. **(Optional) Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify PyTorch and CUDA:**

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 8. Usage Instructions

### Running the Interface

Launch the Gradio web interface with:

```bash
python gradio_interface.py
```

This opens a browser window where you can adjust configuration settings, initialize the model, and start training.

### Training the Model

1. **Configuration Tab:**  
   Adjust parameters like layer depth, chunk length, learning rate, and epoch learning rate (`null_mix_alpha`).  
   Click **"Update Configuration"** and then **"Initialize Model"**.

2. **Training & Generation Tab:**  
   Choose the training mode (WikiText-2 or Story Mode). In Story Mode, upload a text file.  
   Click **"Start Training"** to begin. The interface displays training statistics, loss plots, and null norm plots as the model updates its state.

### Generation and Checkpointing

- **Generation:**  
  Enter an optional prompt and click **"Generate from Current State"** to produce new text.
- **Checkpointing:**  
  Click **"Generate Checkpoint"** to save the current model state along with the null channel history.
- **Pausing/Stopping:**  
  Use the **"Pause Training"** and **"Stop Training"** buttons as needed.

---

## 9. Troubleshooting

- **CUDA Issues:**  
  If you see an error like "Torch not compiled with CUDA enabled," ensure you are running on a CPU-only system. The code checks `torch.cuda.is_available()` before calling CUDA-specific functions.
- **Model Initialization:**  
  Verify that all configuration parameters fall within the allowed ranges.
- **File Upload in Story Mode:**  
  Ensure the uploaded file is a valid text file encoded in a supported format.

---

## 10. License

Project Astarte is released under the GNU Affero General Public License. See the LICENSE file for full details.

---

## 11. Acknowledgments

Project Astarte is inspired by cutting-edge research in stateful neural architectures and dedicated to the legacy of Sir Terry Pratchett. Special thanks to all contributors and supporters who made this project possible.

---

*Warmest Regards,  
Wormwood*

---

This README is designed both as a comprehensive introduction to the project and as a teaching tool to help you (and others) learn the underlying mathematics. Please review, experiment, and feel free to adjust as your understanding grows.
