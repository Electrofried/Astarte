# gradio_interface.py
import gradio as gr
from astarte.web_interface import AstarteInterface, open_tensorboard

def create_interface():
    interface = AstarteInterface()
    
    with gr.Blocks(title="Project Astarte v2.0", css=""" 
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
            with gr.TabItem("Configuration"):
                with gr.Column(elem_classes="container"):
                    with gr.Column(elem_classes="header"):
                        gr.Markdown("# Project Astarte v2.0 Configuration")
                        gr.Markdown("### Set Model and Training Parameters")
                    with gr.Column():
                        layer_depth = gr.Slider(1, 12, value=12, step=1, label="Number of Layers")
                        num_attn_heads = gr.Slider(1, 16, value=8, step=1, label="Number of Attention Heads")
                        embed_size = gr.Number(value=256, label="Embedding Size", precision=0)
                        hidden_size = gr.Number(value=256, label="Hidden Size", precision=0)
                        sequence_length = gr.Number(value=1024, label="Chunk Length (Tokens)", precision=0)
                        max_seq_length = gr.Number(value=4096, label="Max Sequence Length (Tokens)", precision=0)
                        generation_steps = gr.Number(value=20, label="Generation Steps", precision=0)
                        learning_rate = gr.Number(value=1e-3, label="Learning Rate", precision=5)
                        t_start = gr.Number(value=1.0, label="Initial Time (t_start)", precision=1)
                        dt = gr.Number(value=1.0, label="Time Increment (dt)", precision=1)
                        null_mix_alpha = gr.Number(value=0.5, label="Null Mix Alpha", precision=2)
                        
                        update_btn = gr.Button("Update Configuration")
                        config_output = gr.Textbox(label="Configuration Status")
                        
                        init_model_btn = gr.Button("Initialize Model")
                        init_output = gr.Textbox(label="Model Initialization Status")
                
                def update_config_all(layer_depth, generation_steps, sequence_length, max_seq_length, 
                                        null_mix_alpha, num_attn_heads, embed_size, hidden_size,
                                        learning_rate, t_start, dt):
                    interface.config["num_attn_heads"] = int(num_attn_heads)
                    interface.config["embed_size"] = int(embed_size)
                    interface.config["hidden_size"] = int(hidden_size)
                    interface.config["generation_steps"] = int(generation_steps)
                    interface.config["learning_rate"] = float(learning_rate)
                    interface.config["t_start"] = float(t_start)
                    interface.config["dt"] = float(dt)
                    return interface.update_config(layer_depth, generation_steps, sequence_length, max_seq_length, null_mix_alpha)
                
                update_btn.click(fn=update_config_all, 
                                 inputs=[layer_depth, generation_steps, sequence_length, max_seq_length, 
                                         null_mix_alpha, num_attn_heads, embed_size, hidden_size,
                                         learning_rate, t_start, dt],
                                 outputs=[config_output])
                init_model_btn.click(fn=interface.initialize_model, inputs=[], outputs=[init_output])
            
            with gr.TabItem("Training & Generation"):
                with gr.Column(elem_classes="container"):
                    with gr.Column():
                        gr.Markdown("### Training")
                        mode = gr.Radio(choices=["WikiText-2", "Story Mode"], label="Select Training Mode", value="WikiText-2")
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
                        gr.Markdown("### Generated Output")
                        prompt_input = gr.Textbox(label="Optional Prompt", lines=2, placeholder="Enter prompt for generation...")
                        memory_output = gr.Textbox(label="Generated Output", lines=10, interactive=False)
                        generate_btn = gr.Button("Generate from Current State")
                
                def update_story_visibility(choice):
                    return gr.update(visible=choice == "Story Mode")
                mode.change(fn=update_story_visibility, inputs=[mode], outputs=[story_input])
                
                def start_and_toggle_training(mode, story_input):
                    result = interface.start_training(mode, story_input)
                    try:
                        for training_result in result:
                            yield training_result
                    except Exception as e:
                        yield ({"error": str(e)},
                               None,
                               str(e),
                               gr.update(visible=True),
                               gr.update(interactive=False),
                               gr.update(visible=False),
                               gr.update(visible=False))
                
                train_btn.click(fn=start_and_toggle_training, 
                                inputs=[mode, story_input],
                                outputs=[training_stats, loss_plot, memory_output,
                                         train_btn, checkpoint_btn, pause_btn, stop_btn])
                tensorboard_btn.click(fn=open_tensorboard, inputs=[], outputs=[config_output])
                pause_btn.click(fn=interface.pause_resume_training, inputs=[], outputs=[training_stats, pause_btn, checkpoint_btn])
                stop_btn.click(fn=interface.stop_training, inputs=[], outputs=[training_stats, train_btn, pause_btn, stop_btn])
                
                def generate_with_prompt(prompt):
                    return interface.generate_text(prompt)
                generate_btn.click(fn=generate_with_prompt, inputs=[prompt_input], outputs=[memory_output])
                checkpoint_btn.click(fn=interface.generate_checkpoint, inputs=[], outputs=[training_stats])
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)
