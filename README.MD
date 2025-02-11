Below is the updated README with corrected LaTeX formatting for the equations. Please review:

---

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

Project Astarte introduces a new type of stateful neural architecture that integrates periodic state sampling with a central “null channel” mechanism. This design allows the model to continuously update its internal state using fixed-length input chunks (epochs) while maintaining a central, invariant core that is crucial for reconstructing and fine-tuning the model's performance. The dual-phase update of the null channel – involving both a "write-out" and a "write-in" phase – is the key innovation, enabling robust checkpointing and dynamic state evolution.

---

## 2. Project Overview

Project Astarte processes sequential text data by dividing it into fixed-length chunks and updating a multi-channel internal state. The model comprises several key components:

- **Base Update Module (ABPE):** Updates primary state channels \( x_A \) and \( x_B \) and secondary channels \( p_A \) and \( p_B \) using differential equations.
- **Null Channel Mechanism:** Maintains a central null channel \( x_0 \) that is updated in two phases:
  - **Write-Out Phase:**  
    \[
    x_0' = x_0 + \zeta \left( \lvert x_A' - x_B' \rvert - x_0 \right)
    \]
  - **Write-In Phase:**  
    \[
    x_0'' = x_0' + \alpha \left( \bar{x}_0 - x_0' \right)
    \]
- **Attention and Layer Stack:** Aggregates information via attention heads and processes state updates across multiple layers.
- **Epoch Training:** Processes fixed-length token chunks repeatedly and computes loss over epochs.
- **Checkpointing:** Saves model parameters along with the null channel history (aggregated null values) to allow exact state reconstruction.
- **User Interface:** A Gradio-based interface that enables configuration, training, generation, and checkpointing of the model.

---

## 3. Mathematical Foundations

### Base Update Equations

The core state update is performed on five registers:

- \( x_A, x_B \): Primary state channels  
- \( p_A, p_B \): Secondary channels capturing momentum or derivatives  
- \( x_0 \): The null channel (central backbone)

These registers are updated using the following differential equations:

\[
\begin{aligned}
x_A' &= x_A + f_{u_A} \sin(\omega t + \phi) + \lambda \Bigl( p_A - (x_A - x_B) \Bigr), \\
x_B' &= x_B - f_{u_B} \sin(\omega t + \phi) + \lambda \Bigl( p_B - (x_B - x_A) \Bigr), \\
p_A' &= p_A + \eta \Bigl( x_B' - x_A' \Bigr), \\
p_B' &= p_B + \eta \Bigl( x_A' - x_B' \Bigr).
\end{aligned}
\]

These equations are implemented in the **`AutonomicBasePairEncoder`** class in **`astarte/models.py`**.

### Null Channel Write-Out Phase

After the base update, the null channel is updated to capture the difference between the primary channels:

\[
x_0' = x_0 + \zeta \left( \lvert x_A' - x_B' \rvert - x_0 \right)
\]

This equation is part of the forward pass in the **`AutonomicBasePairEncoder.forward`** method.

### Null Injection (Write-In) Phase

The null injection phase re-injects an aggregated null value into the updated null channel. First, a series of null outputs \( x_0^{(i)} \) are normalized using min–max normalization:

\[
\text{Norm}\bigl(x_0^{(i)}\bigr) = \frac{x_0^{(i)} - \min(x_0)}{\max(x_0) - \min(x_0) + \epsilon}
\]

Then, the aggregated null \( \bar{x}_0 \) is computed by averaging:

\[
\bar{x}_0 = \frac{1}{N} \sum_{i=1}^{N} \text{Norm}\bigl(x_0^{(i)}\bigr)
\]

Finally, the null injection phase updates the null channel with a mixing parameter \( \alpha \):

\[
x_0'' = x_0' + \alpha \left( \bar{x}_0 - x_0' \right)
\]

This process is encapsulated in the **`null_cycle`** method of **`AutonomicTokenPredictionModel`** in **`astarte/models.py`**.

### Epoch Training and Fixed-Length Chunks

Input text is divided into fixed-length chunks:

\[
T = \{ t_1, t_2, \dots, t_L \}
\]

Each chunk is processed as:

\[
\text{State}_{\text{new}} = f\Bigl(\text{State}_{\text{old}},\, T,\, x_0\Bigr)
\]

The loss for each chunk is computed, and an epoch is defined as processing \( K \) chunks:

\[
\text{Epoch Loss} = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_k
\]

The **`RollingTextDataset`** in **`astarte/dataset.py`** handles the splitting of input into these chunks.

---

## 4. Code Structure

The repository is organized as follows:

```
project_astarte/
├── astarte/
│   ├── __init__.py
│   ├── models.py              # Neural architecture modules (ABPE, AAAH, AAB, ALS, AIVPL, ATPM)
│   ├── dataset.py             # Implements RollingTextDataset for fixed-length token chunks
│   ├── utils.py               # Helper functions (e.g., detach_state, generate_dream, get_checkpoint_name)
│   └── web_interface.py       # Backend logic for training, generation, and checkpointing
├── gradio_interface.py        # Gradio UI for configuration, training, and generation
└── README.md                  # This document
```

Mapping between the mathematical elements and the code:

| **Mathematical Element**                          | **Description**                                          | **Code Location**                                                |
|---------------------------------------------------|----------------------------------------------------------|------------------------------------------------------------------|
| **Base Update Equations**                         | Updates \( x_A, x_B, p_A, p_B \)                         | `AutonomicBasePairEncoder.forward` in **`astarte/models.py`**      |
| **Null Channel Write-Out Update**                 | \( x_0' = x_0 + \zeta \left( \lvert x_A' - x_B' \rvert - x_0 \right) \) | Also within `AutonomicBasePairEncoder.forward`                  |
| **Null Injection (Write-In) Phase**               | \( x_0'' = x_0' + \alpha \left( \bar{x}_0 - x_0' \right) \) | `AutonomicTokenPredictionModel.null_cycle` in **`astarte/models.py`** |
| **Normalization & Aggregation of Null Outputs**   | Min–max normalization and averaging to compute \( \bar{x}_0 \) | `compute_aggregated_null` in **`astarte/web_interface.py`**       |
| **Chunking and Epoch Training**                   | Splitting input into fixed-length token sequences        | `RollingTextDataset` in **`astarte/dataset.py`**                   |
| **Loss Computation**                              | Cross-entropy loss over chunks                           | Within the training loop in **`astarte/web_interface.py`**         |

---

## 5. Configuration Settings

The behavior of Astarte is governed by a configuration dictionary (located in **`astarte/web_interface.py`**) that includes:

- **`chunk_length`**  
  *Definition:* The length of each token chunk used in training.  
  *Impact:* Defines the granularity of epoch training.

- **`num_layers`**  
  *Definition:* Number of layers in the layer stack.  
  *Impact:* Controls the depth of state propagation.

- **`max_sequence_length`**  
  *Definition:* Maximum allowed sequence length for input data.  
  *Impact:* Prevents inputs from exceeding model capacity.

- **`num_attn_heads`**  
  *Definition:* Number of attention heads per attention block.  
  *Impact:* Influences state aggregation.

- **`embed_size`**  
  *Definition:* Dimensionality of token embeddings.  
  *Impact:* Affects the initial token representation.

- **`hidden_size`**  
  *Definition:* Size of the hidden state channels.  
  *Impact:* Determines internal state capacity.

- **`learning_rate`**  
  *Definition:* Optimizer learning rate.  
  *Impact:* Controls update magnitude during training.

- **`t_start` & `dt`**  
  *Definition:* Time parameters for the sinusoidal components in the state updates.  
  *Impact:* Affect the evolution of the state over layers.

- **`dream_noise_std`** & **`dream_sequence_length`**  
  *Definition:* Parameters for generating new text ("dreaming").  
  *Impact:* Influence the stochasticity and length of generated sequences.

- **`generation_steps`**  
  *Definition:* Number of steps during generation.  
  *Impact:* Determines the duration of a generation cycle.

- **`pause_interval`**  
  *Definition:* Frequency of rest periods during training.  
  *Impact:* Provides opportunities for null channel updates.

- **`checkpoint_dir`**  
  *Definition:* Directory to save checkpoints.  
  *Impact:* Enables model state recovery and fine-tuning.

- **`null_injection_token`**  
  *Definition:* Token used for null injection initialization (defaults to GPT2’s EOS token).  
  *Impact:* Influences initial null channel values.

- **`null_mix_alpha`**  
  *Definition:* Mixing parameter \( \alpha \) for the null injection phase.  
  *Impact:* Serves as the epoch learning rate, controlling how strongly the aggregated null \( \bar{x}_0 \) overrides the current null state.

These settings can be modified via the Gradio UI, directly impacting how the model processes input and updates its state.

---

## 6. DDNA Analogy

Project Astarte’s design is analogous to Dynamic DNA (DDNA):

- **Helix Structure:**  
  Like the double helix with a central backbone (sugar-phosphate backbone) in DNA, Astarte’s architecture is built around a central null channel \( x_0 \), which acts as the invariant core of the internal state.

- **Null Channel as Genetic Code:**  
  The two-phase update (write-out followed by write-in) of \( x_0 \) is akin to the way genetic information is expressed and then re-incorporated via epigenetic mechanisms.

- **Epoch Training as Gene Expression:**  
  Just as genes are repeatedly transcribed to produce proteins, the model processes fixed-length token chunks repeatedly (epochs), with the aggregated null providing a conserved “genetic” signal.

- **Configuration as Genetic Regulation:**  
  Parameters like chunk length, hidden size, and especially \( \text{null_mix_alpha} \) serve as regulators of gene expression, controlling how much influence the invariant null (the “genetic code”) exerts on the overall system.

This analogy helps clarify the innovative data entry and state manipulation mechanisms of Astarte.

---

## 7. Installation and Setup

### Prerequisites
- Python 3.8 or later.
- PyTorch (with CUDA support if desired; CPU-only mode is available).
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
4. Verify PyTorch installation and CUDA support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

---

## 8. Usage Instructions

### Running the Interface

To launch the Gradio interface, run:

```bash
python gradio_interface.py
```

This will start the web-based UI, where you can configure the model, initiate training, generate outputs, and view checkpoint statistics.

### Training the Model

1. **Configuration Tab:**  
   Adjust the model parameters (layer depth, sequence length, learning rate, null mix alpha, etc.) and click **"Update Configuration"**. Then click **"Initialize Model"**.

2. **Training & Generation Tab:**  
   Select a training mode ("WikiText-2" or "Story Mode"). If using "Story Mode," upload a text file. Click **"Start Training"** to begin the training loop. The training loop updates the model state, applies the null injection cycle (using the aggregated normalized null), and displays training statistics, loss plots, and null norm plots.

### Generation and Checkpointing

- **Generation:**  
  Enter an optional prompt and click **"Generate from Current State"** to produce new text based on the current state.

- **Checkpointing:**  
  Click **"Generate Checkpoint"** to save the current model state along with the null channel history. This checkpoint contains both the model parameters and the aggregated null values needed for state reconstruction.

- **Pausing/Stopping:**  
  Use **"Pause Training"** and **"Stop Training"** to manage the training process.

---

## 9. Troubleshooting

- **CUDA Issues:**  
  If you encounter errors like "Torch not compiled with CUDA enabled," ensure that you are running on a system with the appropriate CUDA-enabled PyTorch installation. The code includes checks with `torch.cuda.is_available()` to avoid calling CUDA-specific functions when not supported.

- **Model Initialization:**  
  Verify that all configuration parameters are within the valid ranges (e.g., layer depth between 1 and 12).

- **File Upload (Story Mode):**  
  Ensure that the uploaded files are valid text files with supported encoding.

---

## 10. License

Project Astarte is released under the GNU Affero General Public License. See the LICENSE file for further details.

---

## 11. Acknowledgments

Project Astarte is inspired by advanced research in stateful neural architectures. This project is dedicated to the legacy of Sir Terry Pratchett. Special thanks to all contributors and supporters who have helped bring this project to life.

---

*Warmest Regards,  
Wormwood*
