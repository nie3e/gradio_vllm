import gradio as gr
import os

from gradio_vllm.backend.vllm_client import client
from gradio_vllm.backend import chat

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")


def create_app() -> gr.Blocks:
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    vllm_url = gr.Textbox(
                        label="VLLM base url",
                        value=BASE_URL,
                        interactive=True
                    )
                    vllm_connect_btn = gr.Button(
                        value="Connect",
                        variant="primary"
                    )
            with gr.Column(scale=7):
                connection_status = gr.Label(
                    label="VLLM server connection",
                    show_label=True,
                    value=lambda: client.model_name or "No connection"
                )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    max_completion_tokens = gr.Slider(
                        label="Max completion tokens",
                        minimum=0, maximum=20000, step=1, value=2000
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty",
                        minimum=0.0, maximum=1.0, step=0.1, value=1.0
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0, maximum=2.0, step=0.1, value=0.3
                    )
            with gr.Column(scale=7):
                with gr.Tab("Chat"):
                    with gr.Accordion("System prompt", open=False):
                        system_textbox = gr.Textbox("", label="System prompt")
                    gr.ChatInterface(
                        chat.inference,
                        type="messages",
                        editable=True,
                        show_progress="full",
                        multimodal=False,
                        additional_inputs=[
                            system_textbox,
                            temperature,
                            max_completion_tokens
                        ],
                        concurrency_limit=5,
                    ).chatbot.height = 700
                with gr.Tab("Multimodal"):
                    with gr.Accordion("System prompt", open=False):
                        system_textbox = gr.Textbox("", label="System prompt")
                    gr.ChatInterface(
                        chat.inference,
                        type="messages",
                        editable=True,
                        show_progress="full",
                        multimodal=True,
                        additional_inputs=[
                            system_textbox,
                            temperature,
                            max_completion_tokens,
                            gr.Checkbox(value=True, visible=False,
                                        interactive=False)
                        ],
                        concurrency_limit=5,
                    ).chatbot.height = 700
        vllm_connect_btn.click(
            fn=client.set_connection,
            inputs=vllm_url,
            outputs=connection_status
        )
    return demo
