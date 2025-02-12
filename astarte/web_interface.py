# astarte/web_interface.py
import os
import chardet
import webbrowser
import torch
import gradio as gr
import matplotlib.pyplot as plt

from datetime import datetime
from transformers import GPT2Tokenizer
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset

from astarte.models import AutonomicTokenPredictionModel
from astarte.utils import detach_state, get_checkpoint_name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float64)

class AstarteInterface:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.writer = None
        self.current_state = None
        self.is_training = False
        self.is_paused = False
        self.checkpoint_base = None
        self.optimizer = None
        self.scheduler = None
        self.dataset = None
        self.loss_history = []
        self.text_buffer = []  # For generated text
        self.config = {
            "chunk_length": 1024,
            "num_layers": 12,
            "max_sequence_length": 4096,
            "num_attn_heads": 8,
            "embed_size": 256,
            "hidden_size": 180,
            "learning_rate": 1e-3,
            "t_start": 1.0,
            "dt": 1.0,
            "generation_steps": 20,
            "log_filename": "dream_log.txt",
            "checkpoint_dir": "checkpoints",
            "auto_checkpoint": True,
            "auto_checkpoint_interval": 100,
            "null_injection_token": None,
            "null_mix_alpha": 0.5
        }
        self.setup()

    def setup(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.config["null_injection_token"] is None:
            self.config["null_injection_token"] = self.tokenizer.eos_token_id
        self.tokenizer.model_max_length = self.config["max_sequence_length"]
        os.makedirs(self.config["checkpoint_dir"], exist_ok=True)
        self.checkpoint_base = get_checkpoint_name(self.config)
        self.writer = SummaryWriter(log_dir="runs/astarte_v2")

    def update_config(self, layer_depth, generation_steps, sequence_length, max_seq_length, null_mix_alpha=None):
        assert 1 <= layer_depth <= 12, "Layer depth must be between 1 and 12"
        assert generation_steps > 0, "Generation steps must be positive"
        assert sequence_length > 0, "Sequence length must be positive"
        
        self.config["num_layers"] = int(layer_depth)
        self.config["generation_steps"] = int(generation_steps)
        self.config["chunk_length"] = int(sequence_length)
        self.config["max_sequence_length"] = int(max_seq_length)
        if null_mix_alpha is not None:
            self.config["null_mix_alpha"] = float(null_mix_alpha)
        
        self.tokenizer.model_max_length = self.config["max_sequence_length"]
        self.checkpoint_base = get_checkpoint_name(self.config)
        
        return (f"Configuration updated: layers={layer_depth}, steps={generation_steps}, "
                f"chunk_length={sequence_length}, max_seq={max_seq_length}, α={self.config['null_mix_alpha']}")

    def initialize_model(self):
        if self.model is not None:
            try:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"CUDA error during model cleanup: {str(e)}")
                raise
        self.model = AutonomicTokenPredictionModel(
            self.tokenizer.vocab_size,
            self.config["embed_size"],
            self.config["hidden_size"],
            self.config["num_layers"],
            self.config["num_attn_heads"]
        )
        try:
            self.model = self.model.to(device)
            self.model.double()
        except RuntimeError as e:
            print(f"CUDA error during model initialization: {str(e)}")
            raise

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        return "Model initialized with version 2.0 configuration"

    def create_plot(self, data, title, ylabel):
        plt.figure(figsize=(8, 4))
        plt.plot(data)
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel(ylabel)
        plt.grid(True)
        plot_path = f"temp_{title.lower().replace(' ', '_')}.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def process_long_text(self, text_data):
        token_ids = self.tokenizer.encode(text_data, add_special_tokens=False)
        if len(token_ids) > self.config["max_sequence_length"]:
            print(f"Warning: Input text length ({len(token_ids)}) exceeds maximum sequence length "
                  f"({self.config['max_sequence_length']}). Truncating...")
            token_ids = token_ids[:self.config["max_sequence_length"]]
            truncated_text = self.tokenizer.decode(token_ids)
            print(f"Truncated text length: {len(token_ids)} tokens")
            return truncated_text
        else:
            return text_data

    def read_text_file(self, file_obj):
        try:
            content_bytes = file_obj if isinstance(file_obj, bytes) else file_obj.read()
            result = chardet.detect(content_bytes)
            encoding = result['encoding']
            text_data = content_bytes.decode(encoding)
            print(f"File loaded successfully using {encoding} encoding")
            return text_data
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}. Ensure it is a valid text file.")

    def load_training_data(self, mode, file_obj=None):
        if mode == "WikiText-2":
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
                text_data = "\n\n".join(dataset["text"])
                print("Using WikiText-2 dataset. Total text length:", len(text_data))
            except Exception as e:
                print(f"Error loading WikiText-2: {str(e)}")
                text_data = ("This is a default text used for training Project Astarte. "
                             "It is a sample story that spans multiple beats.")
        else:
            if file_obj is None:
                raise ValueError("No file provided for Story Mode")
            text_data = self.read_text_file(file_obj)
            print("Using custom story text.")
        text_data = self.process_long_text(text_data)
        token_ids = self.tokenizer.encode(text_data, add_special_tokens=False)
        print(f"Tokenized text length: {len(token_ids)} tokens")
        return token_ids

    def update_training_dataset(self, mode, file_obj=None):
        token_ids = self.load_training_data(mode, file_obj)
        from astarte.dataset import RollingTextDataset
        self.dataset = RollingTextDataset(
            token_ids,
            chunk_length=self.config["chunk_length"]
        )
        print("Training dataset updated.")
        return f"Training dataset updated for mode: {mode}"

    def generate_text(self, prompt):
        if self.model is None:
            return "Model not initialized. Please initialize the model before generating text."
        self.model.eval()
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False) if prompt else [self.config["null_injection_token"]]
        generated = prompt_ids[:]
        input_ids = torch.tensor([generated[-self.config["chunk_length"]:]], dtype=torch.long, device=device)
        state = None
        for _ in range(self.config["generation_steps"]):
            logits, state = self.model(input_ids, prev_state=state, t_start=self.config["t_start"], dt=self.config["dt"])
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated[-self.config["chunk_length"]:]], dtype=torch.long, device=device)
        return self.tokenizer.decode(generated)

    def generate_checkpoint(self):
        if self.model is None:
            return {"error": "Model not initialized", "cuda_error": False}
        if not self.is_training:
            return {"error": "Model must be training to generate checkpoint"}
        try:
            checkpoint_path = os.path.join(
                self.config["checkpoint_dir"],
                f"{self.checkpoint_base}_step{self.stats['step']:06d}.pt"
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            torch.save({
                "step": self.stats["step"],
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.stats.get("loss", 0.0),
                "state": self.current_state,
                "config": self.config
            }, checkpoint_path)
            print(f"Checkpoint saved at step {self.stats['step']} -> {checkpoint_path}")
            return {"status": "Checkpoint generated", "path": checkpoint_path}
        except Exception as e:
            print(f"Error during checkpoint generation: {str(e)}")
            return {"error": f"Failed to generate checkpoint: {str(e)}"}

    # NEW: List available checkpoints.
    def list_checkpoints(self):
        checkpoint_dir = self.config["checkpoint_dir"]
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
        return sorted(checkpoints)

    # NEW: Load a checkpoint given its filename.
    def load_checkpoint(self, checkpoint_file):
        checkpoint_path = os.path.join(self.config["checkpoint_dir"], checkpoint_file)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if self.model is None:
                # If no model exists, initialize one using current config.
                self.initialize_model()
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.current_state = checkpoint.get("state", None)
            self.stats = {"step": checkpoint.get("step", 0), "loss": checkpoint.get("loss", 0.0)}
            return f"Checkpoint '{checkpoint_file}' loaded successfully at step {self.stats['step']}."
        except Exception as e:
            return f"Error loading checkpoint: {str(e)}"

    def start_training(self, mode, story_text=None):
        try:
            if self.is_training:
                return ({"status": "Training already in progress"},
                        None,
                        None,
                        gr.update(visible=True),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False))
            if self.model is None:
                self.initialize_model()
            self.loss_history = []
            self.text_buffer = []
            self.is_training = True
            self.stats = {
                "step": 0,
                "loss": 0.0,
                "mode": mode,
                "checkpoint": self.checkpoint_base
            }
            token_ids = self.load_training_data(mode, story_text)
            from astarte.dataset import RollingTextDataset
            self.dataset = RollingTextDataset(
                token_ids,
                chunk_length=self.config["chunk_length"]
            )
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["learning_rate"])
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            memory_output = ""
            # Updated training loop: yield exactly 7 outputs.
            for data in self.dataset:
                if not self.is_training or self.is_paused:
                    break
                input_ids, target = data[:2]
                try:
                    input_ids = input_ids.unsqueeze(0).to(device)
                    target = torch.tensor([target], dtype=torch.long).to(device)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except RuntimeError as e:
                    print(f"CUDA error during data transfer: {str(e)}")
                    raise

                self.optimizer.zero_grad()
                logits, new_state = self.model(
                    input_ids,
                    prev_state=self.current_state,
                    t_start=self.config["t_start"],
                    dt=self.config["dt"]
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                loss = loss_fn(logits, target)
                loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except RuntimeError as e:
                    print(f"CUDA error during gradient clipping: {str(e)}")
                    raise
                self.optimizer.step()
                self.scheduler.step()
                self.stats["loss"] = loss.item()
                self.loss_history.append(loss.item())
                print(f"Step {self.stats['step']}: Loss = {loss.item():.4f}")
                
                self.current_state = detach_state(new_state)
                self.stats["step"] += 1

                if self.config["auto_checkpoint"] and self.stats["step"] % self.config["auto_checkpoint_interval"] == 0:
                    self.generate_checkpoint()

                loss_plot = self.create_plot(self.loss_history, "Training Loss", "Loss")
                yield (self.stats,
                       loss_plot,
                       memory_output,
                       gr.update(visible=False),              # train_btn update
                       gr.update(interactive=True),             # checkpoint_btn update
                       gr.update(visible=True, value="Pause Training"),  # pause_btn update
                       gr.update(visible=True))                 # stop_btn update
        except Exception as e:
            print(f"Training error: {str(e)}")
            self.is_training = False
            return ({"error": str(e)},
                    None,
                    str(e),
                    gr.update(visible=True),
                    gr.update(interactive=False),
                    gr.update(visible=False),
                    gr.update(visible=False))

    def pause_resume_training(self):
        self.is_paused = not self.is_paused
        button_text = "Resume Training" if self.is_paused else "Pause Training"
        return ({"status": "Training paused" if self.is_paused else "Training resumed"},
                gr.update(value=button_text),
                gr.update(interactive=True))

    def stop_training(self, perform_cleanup=True):
        self.is_training = False
        self.is_paused = False
        if perform_cleanup:
            try:
                self.generate_checkpoint()
                if self.model is not None:
                    del self.model
                    self.model = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            except Exception as e:
                print(f"Cleanup error: {str(e)}")
        return ({"status": "Training stopped"},
                gr.update(visible=True, value="Start Training", interactive=True),
                gr.update(visible=False),
                gr.update(visible=False))

def open_tensorboard():
    url = "http://localhost:6006"
    webbrowser.open(url)
    return "TensorBoard opened in browser"
