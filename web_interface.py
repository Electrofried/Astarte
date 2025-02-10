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

web_interface.py: Gradio web interface for Project Astarte
Provides a user-friendly interface for training and using the model.
"""

# Standard library imports
import os
import math
import time
import chardet
import webbrowser
import torch
import gradio as gr
import matplotlib.pyplot as plt

# Project imports
from datetime import datetime
from interface import (
    AutonomicTokenPredictionModel,
    RollingTextDataset,
    generate_dream,
    detach_state
)
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

# Set device and ensure FP64 precision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

def get_checkpoint_name(config):
    """Generate checkpoint name with timestamp and parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    params = f"L{config['num_layers']}_H{config['hidden_size']}_S{config['chunk_length']}"
    return f"checkpoint_{timestamp}_{params}"

def open_tensorboard():
    """Open TensorBoard in default browser"""
    url = "http://localhost:6006"
    webbrowser.open(url)
    return "TensorBoard opened in browser"

class AstarteInterface:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.writer = None
        self.current_state = None
        self.is_training = False
        self.checkpoint_base = None
        self.optimizer = None  # Store optimizer for checkpointing
        self.loss_history = []
        self.null_norm_history = []
        self.config = {
            "chunk_length": 512,
            "num_layers": 6,
            "max_sequence_length": 1000000,
            "num_attn_heads": 4,
            "embed_size": 128,
            "hidden_size": 128,
            "learning_rate": 1e-3,
            "t_start": 1.0,
            "dt": 1.0,
            "dream_noise_std": 0.01,
            "dream_sequence_length": 512,
            "generation_steps": 20,
            "log_filename": "dream_log.txt",
            "pause_interval": 4,
            "checkpoint_dir": "checkpoints",
            "auto_checkpoint": False  # Disabled by default
        }
        self.setup()

    def setup(self):
        """Initialize tokenizer and create directories"""
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.model_max_length = self.config["max_sequence_length"]
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        self.checkpoint_base = get_checkpoint_name(self.config)
        self.writer = SummaryWriter(log_dir="runs/astarte")

    def update_config(self, layer_depth, generation_steps, sequence_length, max_seq_length):
        """Update configuration while maintaining mathematical validity"""
        # Validate parameters
        assert 1 <= layer_depth <= 12, "Layer depth must be between 1 and 12"
        assert generation_steps > 0, "Generation steps must be positive"
        assert sequence_length > 0, "Sequence length must be positive"
        
        self.config["num_layers"] = int(layer_depth)
        self.config["generation_steps"] = int(generation_steps)
        self.config["chunk_length"] = int(sequence_length)
        self.config["dream_sequence_length"] = int(sequence_length)
        self.config["max_sequence_length"] = int(max_seq_length)
        self.tokenizer.model_max_length = self.config["max_sequence_length"]
        self.checkpoint_base = get_checkpoint_name(self.config)
        
        return (f"Configuration updated: layers={layer_depth}, steps={generation_steps}, "
                f"length={sequence_length}, max_seq={max_seq_length}")

    def initialize_model(self):
        """Initialize model with current configuration"""
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()
        
        self.model = AutonomicTokenPredictionModel(
            self.tokenizer.vocab_size,
            self.config["embed_size"],
            self.config["hidden_size"],
            self.config["num_layers"],
            self.config["num_attn_heads"]
        ).to(device)
        self.model.double()  # Ensure FP64 precision
        return "Model initialized with current configuration"

    def create_plot(self, data, title, ylabel):
        """Create a matplotlib plot"""
        plt.figure(figsize=(8, 4))
        plt.plot(data)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.grid(True)
        # Save plot to temporary file
        plot_path = f"temp_{title.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def process_long_text(self, text_data):
        """Process long text by truncating to max sequence length"""
        token_ids = self.tokenizer.encode(text_data, add_special_tokens=False)
        if len(token_ids) > self.config["max_sequence_length"]:
            print(f"Warning: Input text length ({len(token_ids)}) exceeds maximum sequence length "
                  f"({self.config['max_sequence_length']}). Truncating...")
            # Take the first max_sequence_length tokens
            token_ids = token_ids[:self.config["max_sequence_length"]]
            # Decode back to text to show what we're actually using
            truncated_text = self.tokenizer.decode(token_ids)
            print(f"Truncated text length: {len(token_ids)} tokens")
            return truncated_text
        else:
            return text_data

    def read_text_file(self, file_obj):
        """Read text file with encoding detection"""
        try:
            # Read the file content as bytes
            content_bytes = file_obj.read()
            
            # Detect the encoding
            result = chardet.detect(content_bytes)
            encoding = result['encoding']
            
            # Decode the content using detected encoding
            text_data = content_bytes.decode(encoding)
            print(f"File loaded successfully using {encoding} encoding")
            return text_data
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}. "
                           "Please ensure the file is a valid text file.")
        finally:
            file_obj.close()

    def load_training_data(self, mode, file_obj=None):
        """Load training data based on selected mode"""
        if mode == "WikiText-2":
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text_data = "\n\n".join(dataset["text"])
                print("Using WikiText-2 dataset. Total text length:", len(text_data))
            except Exception as e:
                print(f"Error loading WikiText-2: {str(e)}")
                text_data = (
                    "This is a default text used for training Project Astarte. "
                    "It is a sample story that spans multiple beats. "
                    "Each segment flows into the next, but every pause beat the system takes a rest "
                    "to breathe and let go of some past context."
                )
                print("Using fallback text. Length:", len(text_data))
        else:  # Story Mode
            if file_obj is None:
                raise ValueError("No file provided for Story Mode")
            text_data = self.read_text_file(file_obj)
            print("Using custom story text.")
        
        text_data = self.process_long_text(text_data)
        return text_data

    def generate_checkpoint(self):
        """Generate a checkpoint on demand"""
        if self.model is None:
            return {"error": "Model not initialized"}
        
        if not self.is_training:
            return {"error": "Model must be training to generate checkpoint"}

        try:
            checkpoint_path = os.path.join(
                self.config["checkpoint_dir"],
                f"{self.checkpoint_base}_step{self.stats['step']:06d}.pt"
            )
            
            torch.save({
                "step": self.stats["step"],
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "loss": self.stats.get("loss", 0.0),
                "state": self.current_state,
                "config": self.config
            }, checkpoint_path)
            
            return {"status": "Checkpoint generated", "path": checkpoint_path}
        except Exception as e:
            return {"error": f"Failed to generate checkpoint: {str(e)}"}

            
    def start_training(self, mode, story_text=None):
        """Start or resume training"""
        if self.is_training:
            return (
                {"status": "Training already in progress"},
                None,
                None,
                "Training is already in progress",
                gr.update(interactive=True),
                gr.update(interactive=False)
            )
        
        if self.model is None:
            self.initialize_model()

        # Reset history
        self.loss_history = []
        self.null_norm_history = []
        self.is_training = True
        self.stats = {
            "step": 0,
            "loss": 0.0,
            "null_norm": 0.0,
            "mode": mode,
            "checkpoint": self.checkpoint_base
        }

        try:
            # Load training data
            text_data = self.load_training_data(mode, story_text)
            
            # Tokenize text
            token_ids = self.tokenizer.encode(text_data, add_special_tokens=False)
            dataset = RollingTextDataset(
                token_ids,
                chunk_length=self.config["chunk_length"],
                pause_interval=self.config["pause_interval"]
            )

            # Training setup
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["learning_rate"])
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            dream_output = ""

            for input_ids, target, pause in dataset:
                if not self.is_training:
                    break
                
                input_ids = input_ids.unsqueeze(0).to(device)
                target = torch.tensor([target], dtype=torch.long).to(device)
                pause_flag = pause

                if pause_flag:
                    with torch.no_grad():
                        _, new_state = self.model(
                            input_ids,
                            prev_state=self.current_state,
                            pause=True,
                            t_start=self.config["t_start"],
                            dt=self.config["dt"]
                        )
                        # Generate dream
                        dream_tokens = generate_dream(
                            self.model,
                            new_state,
                            sequence_length=self.config["dream_sequence_length"],
                            noise_std=self.config["dream_noise_std"]
                        )
                        dream_output = self.tokenizer.decode(dream_tokens)
                else:
                    self.optimizer.zero_grad()
                    logits, new_state = self.model(
                        input_ids,
                        prev_state=self.current_state,
                        pause=False,
                        t_start=self.config["t_start"],
                        dt=self.config["dt"]
                    )
                    loss = loss_fn(logits, target)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.stats["loss"] = loss.item()
                    self.loss_history.append(loss.item())

                if new_state is not None:
                    self.stats["null_norm"] = new_state[4].norm().item()
                    self.null_norm_history.append(self.stats["null_norm"])
                    self.current_state = detach_state(new_state)
                
                self.stats["step"] += 1

                # Create plots
                loss_plot = self.create_plot(self.loss_history, 'Training Loss', 'Loss')
                null_norm_plot = self.create_plot(self.null_norm_history, 'Null Norm', 'Norm')

                yield (
                    self.stats,
                    loss_plot,
                    null_norm_plot,
                    dream_output,
                    gr.update(interactive=True),
                    gr.update(interactive=False)
                )

        except Exception as e:
            print(f"Training error: {str(e)}")
            self.is_training = False
            return (
                {"error": str(e)},
                None,
                None,
                f"Training error: {str(e)}",
                gr.update(interactive=True),
                gr.update(interactive=True)
            )

    def stop_training(self):
        """Stop training gracefully"""
        self.is_training = False
        return (
            {"status": "Training stopped"},
            gr.update(interactive=True)
        )

    def generate_text(self, prompt):
        """Generate text while preserving mathematical properties"""
        if self.model is None:
            return "Please initialize the model first", {"error": "Model not initialized"}

        self.model.eval()
        generated_text = ""
        stats = {"norms": [], "dreams": []}
        
        with torch.no_grad():
            # Initialize state with prompt
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_ids) < self.config["chunk_length"]:
                padded = prompt_ids + [0] * (self.config["chunk_length"] - len(prompt_ids))
                input_ids = torch.tensor([padded], dtype=torch.long, device=device)
            else:
                input_ids = torch.tensor([prompt_ids[-self.config["chunk_length"]:]], dtype=torch.long, device=device)
            
            logits, state = self.model(
                input_ids,
                prev_state=None,
                pause=False,
                t_start=self.config["t_start"],
                dt=self.config["dt"]
            )
            generated_ids = prompt_ids[:]

            # Generate text
            for step in range(self.config["generation_steps"]):
                if (step + 1) % 4 == 0:  # Rest period
                    dummy = torch.zeros((1, 1), dtype=torch.long, device=device)
                    logits, state = self.model(
                        dummy,
                        prev_state=state,
                        pause=True,
                        t_start=self.config["t_start"],
                        dt=self.config["dt"]
                    )
                    # Generate dream
                    dream_tokens = generate_dream(
                        self.model,
                        state,
                        sequence_length=self.config["chunk_length"],
                        noise_std=self.config["dream_noise_std"]
                    )
                    stats["dreams"].append({
                        "step": step,
                        "text": self.tokenizer.decode(dream_tokens)
                    })
                else:
                    if len(generated_ids) == 0:
                        last_token = self.tokenizer.encode(" ")[0]
                    else:
                        last_token = generated_ids[-1]
                    
                    input_tensor = torch.tensor([[last_token]], dtype=torch.long, device=device)
                    logits, state = self.model(
                        input_tensor,
                        prev_state=state,
                        pause=False,
                        t_start=self.config["t_start"],
                        dt=self.config["dt"]
                    )
                    probs = torch.softmax(logits, dim=-1)
                    pred_token = torch.multinomial(probs, num_samples=1).item()
                    generated_ids.append(pred_token)
                
                if state is not None:
                    stats["norms"].append({
                        "step": step,
                        "value": state[4].norm().item()
                    })
                
                generated_text = self.tokenizer.decode(generated_ids)
                stats["current_text"] = generated_text
                stats["decoded_text"] = self.tokenizer.decode(generated_ids)
                yield generated_text, stats

def create_interface():
    """Create the Gradio interface"""
    interface = AstarteInterface()
    
    with gr.Blocks(title="Astarte", css="""
        .container { margin: 0 auto; max-width: 1200px; padding: 20px; }
        .header { text-align: center; margin-bottom: 40px; }
        .project-info { 
            max-height: 200px;
            overflow-y: auto;
            margin: 20px 0;
            padding: 20px;
            background-color: #00843d;
            border-radius: 8px;
            line-height: 1.6;
        }
        .footer {
            text-align: center;
            margin-top: 20px;
            padding: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.9em;
            color: #666;
        }
    """) as app:
        with gr.Column(elem_classes="container"):
            # Header
            with gr.Column(elem_classes="header"):
                gr.Markdown("# Project Astarte")
                gr.Markdown("### A Stateful Neural Architecture with Periodic State Sampling")
            
            # Project Info
            with gr.Column(elem_classes="project-info"):
                gr.Markdown("""
                Astarte introduces a novel stateful neural architecture that processes text sequentially 
                with periodic state sampling and rest periods. The model maintains five internal state registers 
                that are updated via an Autonomic Base Pair Encoder (ABPE) and aggregated through an Autonomic 
                Aggregated Attention Head (AAAH).

                During periodic rest periods—occurring every fourth step—the system suspends new input to 
                propagate its state and perform state sampling, enabling the network to consolidate long-range 
                dependencies while selectively updating its internal state.

                Key Features:
                - Five internal state registers for robust state management
                - Periodic rest periods for state consolidation
                - Autonomic Base Pair Encoder for state updates
                - Aggregated Attention Head for state processing
                - Support for both WikiText-2 and custom story input
                """)

            with gr.Row():
                # Left column: Configuration and Training
                with gr.Column():
                    gr.Markdown("### Model Configuration")
                    layer_depth = gr.Slider(
                        minimum=1,
                        maximum=12,
                        value=6,
                        step=1,
                        label="Layer Depth"
                    )
                    generation_steps = gr.Number(
                        value=20,
                        label="Generation Steps",
                        precision=0
                    )
                    sequence_length = gr.Number(
                        value=512,
                        label="Sequence Length",
                        precision=0
                    )
                    max_seq_length = gr.Number(
                        value=1000000,
                        minimum=1000,
                        maximum=10000000,
                        label="Max Sequence Length",
                        precision=0
                    )
                    update_btn = gr.Button("Update Configuration")
                    config_output = gr.Textbox(label="Configuration Status")
                    
                    init_model_btn = gr.Button("Initialize Model")
                    init_output = gr.Textbox(label="Initialization Status")

                    gr.Markdown("### Training")
                    mode = gr.Radio(
                        choices=["WikiText-2", "Story Mode"],
                        label="Training Mode",
                        value="WikiText-2"
                    )
                    
                    story_input = gr.File(
                        label="Upload Text File",
                        visible=False,
                        file_types=[".txt"],
                        file_count="single",
                        type="binary"
                    )
                    
                    with gr.Row():
                        start_btn = gr.Button("Start Training")
                        stop_btn = gr.Button("Stop Training", interactive=False)
                    
                    with gr.Row():
                        tensorboard_btn = gr.Button("Open TensorBoard")
                        checkpoint_btn = gr.Button("Generate Checkpoint", interactive=False)

                # Right column: Visualization and Generation
                with gr.Column():
                    gr.Markdown("### Training Progress")
                    training_stats = gr.JSON(label="Training Statistics")
                    with gr.Row():
                        loss_plot = gr.Image(label="Loss Plot")
                        null_norm_plot = gr.Image(label="Null Norm Plot")
                    dream_output = gr.Textbox(
                        label="Dream Output",
                        lines=3
                    )

                    gr.Markdown("### Text Generation")
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        lines=2,
                        placeholder="Enter text prompt..."
                    )
                    generate_btn = gr.Button("Generate")
                    generation_output = gr.Textbox(
                        label="Generated Text",
                        lines=5
                    )
                    generation_stats = gr.JSON(label="Generation Stats (Norms & Dreams)")

            # Footer
            with gr.Column(elem_classes="footer"):
                gr.Markdown("GNU Terry Pratchett")
                gr.Markdown("In memory of Sir Terry Pratchett and the Discworld series that inspired this project.")

        # Show/hide story input based on mode
        def update_story_visibility(choice):
            return gr.update(visible=choice == "Story Mode")
        
        mode.change(
            update_story_visibility,
            inputs=[mode],
            outputs=[story_input]
        )

        # Wire up events
        update_btn.click(
            interface.update_config,
            inputs=[layer_depth, generation_steps, sequence_length, max_seq_length],
            outputs=[config_output]
        )
        
        init_model_btn.click(
            interface.initialize_model,
            inputs=[],
            outputs=[init_output]
        )
        
        start_btn.click(
            interface.start_training,
            inputs=[mode, story_input],
            outputs=[
                training_stats,
                loss_plot,
                null_norm_plot,
                dream_output,
                stop_btn,
                start_btn
            ]
        ).then(
            lambda: gr.update(interactive=True),
            inputs=[],
            outputs=[checkpoint_btn]
        )
        
        stop_btn.click(
            interface.stop_training,
            inputs=[],
            outputs=[training_stats, start_btn]
        ).then(
            lambda: gr.update(interactive=False),
            inputs=[],
            outputs=[checkpoint_btn]
        )
        
        tensorboard_btn.click(
            open_tensorboard,
            inputs=[],
            outputs=[config_output]
        )
        
        checkpoint_btn.click(
            interface.generate_checkpoint,
            inputs=[],
            outputs=[training_stats]
        )
        
        generate_btn.click(
            interface.generate_text,
            inputs=[prompt_input],
            outputs=[generation_output, generation_stats]
        )
    
    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False             # Don't create public URL by default
    )