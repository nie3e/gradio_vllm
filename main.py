import base64
import os
from itertools import cycle
from typing import Generator

from openai import OpenAI
import gradio as gr

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def get_model_name() -> str:
    model_name = client.models.list().data[0].id
    return model_name


def inference(
        message: str, history: list[str], system_prompt: str,
        repetition_penalty: float,
        temperature: float, max_completion_tokens: int
) -> Generator[str, None, None]:
    flat_history = [
        h for hs in history for h in hs
    ]
    messages = flat_history + [message]
    messages_dict: list[dict] = [
        {"role": u, "content": msg}
        for u, msg in zip(cycle(["user", "assistant"]), messages)
    ]

    if system_prompt:
        messages_dict = ([{"role": "system", "content": system_prompt}] +
                         messages_dict)

    stream = client.chat.completions.create(
        model=get_model_name(),
        messages=messages_dict,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stream=True
    )
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


def multimodal_inference(
    message: dict, history: list[str], system_prompt: str,
    repetition_penalty: float,
    temperature: float, max_completion_tokens: int
) -> Generator[str, None, None]:
    messages: list[dict] = []
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    images = []
    for couple in history:
        if type(couple[0]) is tuple:
            images += couple[0]
        elif couple[0][1]:
            messages.append({
                "role": "user",
                "content":
                    [{"type": "text", "text": couple[0][1]}] +
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_base64(path)}
                        }
                        for path in images
                    ]
            })
            messages.append({"role": "assistant", "content": couple[1]})
            images = []

    messages.append({
        "role": "user",
        "content":
            [
                {"type": "text", "text": message["text"]}
            ] +
            [
                {
                    "type": "image_url",
                    "image_url": {"url": image_to_base64(file)}
                }
                for file in message["files"]
            ]
    })
    stream = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stream=True
    )
    partial_message = ""
    for chunk in stream:
        partial_message += (chunk.choices[0].delta.content or "")
        yield partial_message


def main() -> int:
    """Entry point function for the web app."""
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group():
                    max_completion_tokens = gr.Slider(
                        label="Max completion tokens",
                        minimum=1, maximum=20000, step=1, value=1000
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty",
                        minimum=0.0, maximum=1.0, step=0.1, value=1.0
                    )
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1, maximum=3.0, step=0.1, value=0.3
                    )
            with gr.Column(scale=7):
                with gr.Tab("Chat"):
                    gr.ChatInterface(
                        inference,
                        description=get_model_name(),
                        show_progress="full",
                        multimodal=False,
                        additional_inputs=[
                            gr.Textbox("", label="System prompt"),
                            repetition_penalty, temperature,
                            max_completion_tokens
                        ],
                        concurrency_limit=5
                    ).chatbot.height = 700
                with gr.Tab("Multimodal"):
                    gr.ChatInterface(
                        multimodal_inference,
                        show_progress="full",
                        multimodal=True,
                        additional_inputs=[
                            gr.Textbox("", label="System prompt"),
                            repetition_penalty, temperature,
                            max_completion_tokens
                        ],
                        concurrency_limit=5
                    ).chatbot.height = 700
    demo.queue(max_size=10).launch(server_name="0.0.0.0")

    return 0


if __name__ == "__main__":
    exit(main())
