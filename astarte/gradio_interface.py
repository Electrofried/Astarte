# gradio_interface.py
import gradio as gr
from astarte.web_interface import AstarteInterface, open_tensorboard

def create_interface():
    interface = AstarteInterface()
    
    with gr.Blocks(title="Project Astarte", css="""
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
        with gr.Tabs():
            # Tab 1: Configuration Settings
            with gr.TabItem("Configuration"):
                with gr.Column(elem_classes="container"):
                    with gr.Column(elem_classes="header"):
                        gr.Markdown("# Project Astarte Configuration")
                        gr.Markdown("### Set Up Model Parameters")
                    
                    with gr.Column():
                        layer_depth = gr.Slider(1, 12, value=6, step=1, label="Layer Depth")
                        generation_steps = gr.Number(value=20, label="Sample Steps", precision=0)
                        sequence_length = gr.Number(value=512, label="Chunk (Sequence) Length", precision=0)
                        max_seq_length = gr.Number(value=2048, label="Max Sequence Length", precision=0)
                        null_mix_alpha = gr.Number(value=0.5, label="Epoch Learning Rate (alpha)", precision=2)
                        num_attn_heads = gr.Slider(1, 16, value=4, step=1, label="Number of Attention Heads")
                        embed_size = gr.Number(value=512, label="Embedding Size", precision=0)
                        hidden_size = gr.Number(value=128, label="Hidden Size", precision=0)
                        learning_rate = gr.Number(value=1e-3, label="Learning Rate", precision=5)
                        t_start = gr.Number(value=1.0, label="t_start (Initial Time)", precision=1)
                        dt = gr.Number(value=1.0, label="dt (Time Increment)", precision=1)
                        dream_noise_std = gr.Number(value=0.01, label="Dream Noise Std", precision=3)
                        dream_sequence_length = gr.Number(value=512, label="Dream Sequence Length", precision=0)
                        pause_interval = gr.Number(value=4, label="Pause Interval (Steps)", precision=0)
                        
                        update_btn = gr.Button("Update Configuration")
                        config_output = gr.Textbox(label="Configuration Status")
                        
                        init_model_btn = gr.Button("Initialize Model")
                        init_output = gr.Textbox(label="Initialization Status")
                
                # Function to update all configuration settings.
                def update_config_all(layer_depth, generation_steps, sequence_length, max_seq_length, 
                                        null_mix_alpha, num_attn_heads, embed_size, hidden_size,
                                        learning_rate, t_start, dt, dream_noise_std, dream_sequence_length, pause_interval):
                    # Update additional config parameters directly.
                    interface.config["num_attn_heads"] = int(num_attn_heads)
                    interface.config["embed_size"] = int(embed_size)
                    interface.config["hidden_size"] = int(hidden_size)
                    interface.config["learning_rate"] = float(learning_rate)
                    interface.config["t_start"] = float(t_start)
                    interface.config["dt"] = float(dt)
                    interface.config["dream_noise_std"] = float(dream_noise_std)
                    interface.config["dream_sequence_length"] = int(dream_sequence_length)
                    interface.config["pause_interval"] = int(pause_interval)
                    # Update primary configuration.
                    return interface.update_config(layer_depth, generation_steps, sequence_length, max_seq_length, null_mix_alpha)
                
                update_btn.click(fn=update_config_all, 
                                 inputs=[layer_depth, generation_steps, sequence_length, max_seq_length, 
                                         null_mix_alpha, num_attn_heads, embed_size, hidden_size,
                                         learning_rate, t_start, dt, dream_noise_std, dream_sequence_length, pause_interval],
                                 outputs=[config_output])
                init_model_btn.click(fn=interface.initialize_model, inputs=[], outputs=[init_output])
            
            # Tab 2: Training and Generation
            with gr.TabItem("Training & Generation"):
                with gr.Column(elem_classes="container"):
                    with gr.Column():
                        gr.Markdown("### Training")
                        mode = gr.Radio(choices=["WikiText-2", "Story Mode"], label="Training Mode", value="WikiText-2")
                        story_input = gr.File(label="Upload Text File", visible=False, file_types=[".txt"], file_count="single", type="binary")
                        with gr.Row():
                            train_btn = gr.Button("Start Training", variant="primary")
                            pause_btn = gr.Button("Pause Training", visible=False)
                            stop_btn = gr.Button("Stop Training", visible=False)
                        with gr.Row():
                            tensorboard_btn = gr.Button("Open TensorBoard")
                            checkpoint_btn = gr.Button("Generate Checkpoint", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("### Training Progress")
                        training_stats = gr.JSON(label="Training Statistics")
                        with gr.Row():
                            loss_plot = gr.Image(label="Loss Plot")
                            null_norm_plot = gr.Image(label="Null Norm Plot")
                        gr.Markdown("### Memory Output")
                        prompt_input = gr.Textbox(label="Optional Prompt", lines=2, placeholder="Enter prompt for generation...")
                        memory_output = gr.Textbox(label="Generated Memory Output", lines=10, interactive=False)
                        generate_btn = gr.Button("Generate from Current State")
                
                # UI logic to update story mode visibility.
                def update_story_visibility(choice):
                    return gr.update(visible=choice == "Story Mode")
                mode.change(fn=update_story_visibility, inputs=[mode], outputs=[story_input])
                
                # Training loop function.
                def start_and_toggle_training(mode, story_input):
                    result = interface.start_training(mode, story_input)
                    try:
                        for training_result in result:
                            yield training_result
                    except Exception as e:
                        yield ({"error": str(e)}, None, None, str(e),
                               gr.update(visible=True), gr.update(interactive=False),
                               gr.update(visible=False), gr.update(visible=False))
                
                train_btn.click(fn=start_and_toggle_training, 
                                inputs=[mode, story_input],
                                outputs=[training_stats, loss_plot, null_norm_plot, memory_output,
                                         train_btn, checkpoint_btn, pause_btn, stop_btn])
                tensorboard_btn.click(fn=open_tensorboard, inputs=[], outputs=[config_output])
                pause_btn.click(fn=interface.pause_resume_training, inputs=[], outputs=[training_stats, pause_btn, checkpoint_btn])
                stop_btn.click(fn=interface.stop_training, inputs=[], outputs=[training_stats, train_btn, pause_btn, stop_btn])
                
                def generate_with_checkpoint(prompt):
                    if interface.is_training and not interface.is_paused:
                        interface.pause_resume_training()
                    return interface.generate_from_state(interface.current_state, prompt)
                
                generate_btn.click(fn=generate_with_checkpoint, inputs=[prompt_input], outputs=[memory_output])
                checkpoint_btn.click(fn=interface.generate_checkpoint, inputs=[], outputs=[training_stats])
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
