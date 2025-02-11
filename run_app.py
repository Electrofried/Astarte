# run_app.py
from astarte.gradio_interface import create_interface

if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Default Gradio port
        share=False              # Generate a public shareable link
    )
