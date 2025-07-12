import base64
import os
from typing import Generator, Iterable

from gradio.components.chatbot import ChatMessage
from openai import OpenAI
import gradio as gr

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
BASE_URL = "http://series2.rav:8000/v1"

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def get_model_name() -> str:
    model_name = client.models.list().data[0].id
    return model_name


def parse_stream(
    content_stream: Iterable,
) -> Generator[list[ChatMessage], None, None]:
    partial_message = ""
    think_mode = False
    answer_started = False
    messages_chat = []

    for chunk in content_stream:
        if chunk.choices[0].delta.content is None:
            continue

        partial_message += chunk.choices[0].delta.content

        if not think_mode and not answer_started:
            if partial_message.startswith("<think>"):
                think_mode = True
                partial_message = partial_message[7:]
                messages_chat.append(
                    ChatMessage(
                        role="assistant",
                        content=partial_message,
                        metadata={"title": "⏳Thinking"},
                    )
                )
            elif (len(partial_message) >= 7
                  or not partial_message.startswith("<")):
                answer_started = True
                messages_chat.append(
                    ChatMessage(
                        role="assistant",
                        content=partial_message,
                    )
                )

        if think_mode and not answer_started and "</think>" in partial_message:
            think_mode = False
            answer_started = True
            think_content, _, answer_content = partial_message.partition(
                "</think>")

            if messages_chat:
                messages_chat[-1].content = think_content

            if answer_content:
                if answer_content.startswith("<answer>"):
                    answer_content = answer_content[8:]
                if answer_content.endswith("</answer>"):
                    answer_content = answer_content[:-9]

            messages_chat.append(
                ChatMessage(
                    role="assistant",
                    content=answer_content,
                )
            )
            partial_message = answer_content

        if messages_chat and answer_started:
            current_content = partial_message
            _, _, answer_content = current_content.partition("<answer>")

            if "</answer>" in answer_content:
                answer_content = answer_content.replace("</answer>", "")
            messages_chat[-1].content = answer_content or current_content
        elif messages_chat:
            if not (
                think_mode and not answer_started
                and "</think>" in partial_message
            ):
                messages_chat[-1].content = partial_message

        yield messages_chat


def inference(
    message: str,
    history: list[dict],
    system_prompt: str,
    temperature: float,
    max_completion_tokens: int,
) -> Generator[list, None, None]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages += [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if not msg.get("metadata")
    ]
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stream=True,
    )

    yield from parse_stream(stream)
    return


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
    stream = client.chat.completions.create(  # noqa
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
                        minimum=0.0, maximum=2.0, step=0.1, value=0.3
                    )
            with gr.Column(scale=7):
                with gr.Tab("Chat"):
                    with gr.Accordion("System prompt", open=False):
                        system_textbox = gr.Textbox("", label="System prompt")
                    gr.ChatInterface(
                        inference,
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
                        multimodal_inference,
                        show_progress="full",
                        multimodal=True,
                        additional_inputs=[
                            system_textbox,
                            repetition_penalty, temperature,
                            max_completion_tokens
                        ],
                        concurrency_limit=5
                    ).chatbot.height = 700
    demo.queue(max_size=10).launch()

    return 0


if __name__ == "__main__":
    exit(main())
