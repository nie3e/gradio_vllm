import gradio as gr

from gradio_vllm.main import get_model_name
from gradio_vllm.backend import chat, multimodal


def create_app() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    max_completion_tokens = gr.Slider(
                        label="Max completion tokens",
                        minimum=1, maximum=20000, step=1, value=2000
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
                        description=get_model_name(),
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
                        multimodal.inference,
                        type="messages",
                        editable=True,
                        description=get_model_name(),
                        show_progress="full",
                        multimodal=True,
                        additional_inputs=[
                            system_textbox,
                            temperature,
                            max_completion_tokens,
                        ],
                        concurrency_limit=5,
                    ).chatbot.height = 700
    return demo
