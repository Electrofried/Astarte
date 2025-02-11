# Project Astarte  
## A Stateful Neural Architecture with Periodic State Sampling

*Warmest Regards,  
Wormwood*

---

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Mathematical Foundations](#mathematical-foundations)
    - [Base Update Equations](#base-update-equations)
    - [Null Channel Write-Out Phase](#null-channel-write-out-phase)
    - [Null Injection (Write-In) Phase](#null-injection-write-in-phase)
    - [Epoch Training and Fixed-Length Chunks](#epoch-training-and-fixed-length-chunks)
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

Project Astarte introduces a new type of stateful neural architecture that integrates periodic state sampling with a central "null channel" mechanism. This design allows the model to continuously update its internal state using fixed-length input chunks (epochs) while maintaining a central, invariant core that is crucial for reconstructing and fine-tuning the model's performance. The dual-phase update of the null channel – involving both a "write-out" and a "write-in" phase – is the key innovation, enabling robust checkpointing and dynamic state evolution.

---

## 2. Project Overview

Project Astarte is designed to process sequential text data by dividing it into fixed-length chunks and updating a multi-channel internal state. The model comprises several key components:

- **Base Update Module (ABPE):** Updates primary state channels (\( x_A, x_B \)) and secondary channels (\( p_A, p_B \)) using differential equations.
- **Null Channel Mechanism:** Maintains a central null channel (\( x_0 \)) that is updated in two phases:
  - **Write-Out Phase:** \( x_0' = x_0 + \zeta \bigl(|x_A' - x_B'| - x_0\bigr) \)
  - **Write-In Phase:** \( x_0'' = x_0' + \alpha \bigl(\bar{x}_0 - x_0'\bigr) \)
- **Attention and Layer Stack:** Aggregates information via attention heads and processes state updates across multiple layers.
- **Epoch Training:** Processes fixed-length token chunks repeatedly and computes loss over epochs.
- **Checkpointing:** Saves model parameters along with the null channel history (aggregated null values) to allow exact state reconstruction.
- **User Interface:** A Gradio-based interface that enables configuration, training, generation, and checkpointing of the model.

---

## 3. Mathematical Foundations

### Base Update Equations

The core state update is performed on five state registers:

- **\( x_A, x_B \):** Primary state channels.
- **\( p_A, p_B \):** Secondary channels capturing momentum or derivatives.
- **\( x_0 \):** The null channel (central backbone).

These registers are updated using the following differential equations:

\[
\begin{aligned}
x_A' &= x_A + f_{u_A}\,\sin(\omega t + \phi) + \lambda\,\Bigl(p_A - (x_A - x_B)\Bigr) \\
x_B' &= x_B - f_{u_B}\,\sin(\omega t + \phi) + \lambda\,\Bigl(p_B - (x_B - x_A)\Bigr) \\
p_A' &= p_A + \eta\,(x_B' - x_A') \\
p_B' &= p_B + \eta\,(x_A' - x_B')
\end{aligned}
\]

*These equations are implemented in the* **`AutonomicBasePairEncoder`** *class in* **`astarte/models.py`**.

### Null Channel Write-Out Phase

After the base update, the null channel is updated using:

\[
x_0' = x_0 + \zeta\,\Bigl(\lvert x_A' - x_B' \rvert - x_0\Bigr)
\]

*This equation is also part of the* **`AutonomicBasePairEncoder.forward`** *method and influences how the null channel reflects differences in the primary channels.*

### Null Injection (Write-In) Phase

A key innovation in Astarte is the null injection phase. During training, the model stores a series of null channel outputs \( x_0^{(i)} \). These outputs are normalized and averaged to yield an aggregated null value \( \bar{x}_0 \):

1. **Normalization:**
   \[
   \text{Norm}\bigl(x_0^{(i)}\bigr) = \frac{x_0^{(i)} - \min(x_0)}{\max(x_0) - \min(x_0) + \epsilon}
   \]
2. **Aggregation:**
   \[
   \bar{x}_0 = \frac{1}{N} \sum_{i=1}^{N} \text{Norm}\bigl(x_0^{(i)}\bigr)
   \]

Then the null injection phase re-injects this aggregated null value with a mixing parameter \( \alpha \) (configured as `null_mix_alpha`):

\[
x_0'' = x_0' + \alpha\,\Bigl(\bar{x}_0 - x_0'\Bigr)
\]

*This process is encapsulated in the* **`null_cycle`** *method of* **`AutonomicTokenPredictionModel`** *in* **`astarte/models.py`**.

### Epoch Training and Fixed-Length Chunks

The input text is segmented into chunks of fixed length \( L \):

\[
T = \{ t_1, t_2, \dots, t_L \}
\]

Each chunk is processed through the model to update the internal state:

\[
\text{State}_{\text{new}} = f\Bigl(\text{State}_{\text{old}},\, T,\, x_0\Bigr)
\]

The loss for each chunk is computed using cross-entropy (or another loss function), and an epoch is defined as processing a series of such chunks.

*Chunking is handled by the* **`RollingTextDataset`** *class in* **`astarte/dataset.py`**.

---

## 4. Code Structure

The repository is organized as follows:

```
project_astarte/
├── astarte/
│   ├── __init__.py
│   ├── models.py              # Contains the neural architecture modules (ABPE, AAAH, AAB, ALS, AIVPL, ATPM)
│   ├── dataset.py             # Implements the RollingTextDataset for fixed-length token chunks
│   ├── utils.py               # Helper functions (e.g., detach_state, generate_dream, get_checkpoint_name)
│   └── web_interface.py       # Backend logic for training, generation, and checkpointing
├── gradio_interface.py        # Gradio UI that ties together configuration, training, and generation
└── README.md                  # This document
```

Each module is carefully mapped to parts of the mathematical model:
- **`models.py`**: Implements the core equations.
- **`dataset.py`**: Manages input segmentation.
- **`utils.py`**: Provides utility functions.
- **`web_interface.py`**: Contains the training loop, null injection, checkpointing, and state aggregation.
- **`gradio_interface.py`**: Provides the web-based UI for configuration and interaction.

---

## 5. Configuration Settings

The model’s behavior is governed by a configuration dictionary (found in **`astarte/web_interface.py`**), which includes:

- **`chunk_length`**: Length of each token chunk used in training.
- **`num_layers`**: Number of layers in the model’s layer stack.
- **`max_sequence_length`**: Maximum allowed sequence length for input data.
- **`num_attn_heads`**: Number of attention heads per attention block.
- **`embed_size`**: Dimensionality of token embeddings.
- **`hidden_size`**: Size of the hidden state channels.
- **`learning_rate`**: Learning rate for the optimizer.
- **`t_start` & `dt`**: Time parameters influencing the sinusoidal update in the state equations.
- **`dream_noise_std`** & **`dream_sequence_length`**: Parameters used during token generation ("dreaming").
- **`generation_steps`**: Number of generation steps performed.
- **`pause_interval`**: Interval (in training steps) for entering a rest period.
- **`checkpoint_dir`**: Directory for saving model checkpoints.
- **`null_injection_token`**: The token used for initializing null injection (defaults to GPT2’s EOS token).
- **`null_mix_alpha`**: The mixing parameter \( \alpha \) used during the null injection phase; effectively an epoch learning rate that controls the degree to which the aggregated null \( \bar{x}_0 \) overrides the current null state.

These parameters can be updated via the Gradio UI in the Configuration tab, which directly influences how the model processes input and evolves its state.

---

## 6. DDNA Analogy

Project Astarte draws an analogy to DDNA (Dynamic DNA) in its design:

- **Helix Structure:**  
  Just as DNA is built around a double helix with a central backbone (sugar-phosphate backbone), Astarte’s architecture centers on the null channel \( x_0 \), which acts as the invariant core or “backbone” of the internal state.

- **Null Channel as Genetic Code:**  
  The null channel undergoes a two-phase update:
  - **Write-Out Phase:** The channel is updated based on the difference between primary state channels.
  - **Write-In Phase:** An aggregated, normalized version of past null outputs is injected back, similar to how genetic information is re-read and expressed.
  
- **Epoch Training and Repetitive Expression:**  
  In DNA, gene expression occurs in cycles, and the same sequence (genetic code) is repeatedly read to produce proteins. In Astarte, fixed-length chunks (epochs) are processed repeatedly, with the aggregated null acting like a conserved genetic code that influences the network’s behavior over time.

- **Configuration as Genetic Regulation:**  
  The various configuration settings (such as chunk length, hidden size, and especially \( \text{null_mix_alpha} \)) regulate how strongly the invariant null channel impacts the state update—analogous to genetic regulatory mechanisms controlling gene expression.

This analogy helps explain the innovative data entry and state update process of Astarte.

---

## 7. Installation and Setup

### Prerequisites
- Python 3.8 or later.
- PyTorch (ensure CUDA support if desired; otherwise, CPU-only mode is available).
- Other dependencies:
  - Transformers
  - Gradio
  - Datasets
  - Matplotlib
  - Chardet

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/project_astarte.git
   cd project_astarte
   ```
2. (Optional) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Verify your PyTorch installation and check for CUDA support:
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 8. Usage Instructions

### Running the Interface

To launch the Gradio interface, run:

```bash
python gradio_interface.py
```

This command starts the web-based UI, allowing you to configure the model, initiate training, generate outputs, and view checkpoint statistics.

### Training the Model

1. **Configuration Tab:**  
   Adjust model parameters such as layer depth, chunk length, learning rate, and the epoch learning rate (`null_mix_alpha`).  
   Click **"Update Configuration"** to apply settings, then **"Initialize Model"** to set up the model.

2. **Training & Generation Tab:**  
   Select a training mode ("WikiText-2" or "Story Mode"). In "Story Mode", upload a text file.  
   Click **"Start Training"** to begin the training loop. The training loop updates the model state, performs null injection cycles (using the aggregated normalized null), and displays training statistics, loss plots, and null norm plots.

### Generation and Checkpointing

- **Generation:**  
  In the Training & Generation tab, enter an optional prompt and click **"Generate from Current State"** to produce new text based on the current state.
  
- **Checkpointing:**  
  Click **"Generate Checkpoint"** to save the current model state along with the null channel history. This checkpoint contains both model parameters and the aggregated null values needed to reconstruct the internal state.

- **Pausing/Stopping:**  
  Use **"Pause Training"** and **"Stop Training"** to manage the training process.

---

## 9. Troubleshooting

- **CUDA Issues:**  
  If you encounter "Torch not compiled with CUDA enabled," ensure you’re running on a CPU-only system. The code now checks `torch.cuda.is_available()` before calling CUDA-specific functions.
  
- **Model Initialization:**  
  If the model fails to initialize, verify that all configuration parameters are within the valid ranges (e.g., layer depth between 1 and 12).

- **File Upload (Story Mode):**  
  Ensure that uploaded files are valid text files encoded in a supported format.

---

## 10. License

Project Astarte is distributed under the GNU Affero General Public License. See the LICENSE file for details.

---

## 11. Acknowledgments

Project Astarte is inspired by cutting-edge research in stateful neural architectures and is dedicated to the legacy of storytellers in all of us. Special thanks to all contributors and supporters who made this project possible.

---

*Warmest Regards,  
Wormwood*