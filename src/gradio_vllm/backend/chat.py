from typing import Generator, Any

from gradio_vllm.backend.helper import (
    parse_stream, prepare_chat_messages, prepare_multimodal_messages
)
from gradio_vllm.backend.vllm_client import client


def inference(
    message: str | dict[str, Any],
    history: list[dict],
    system_prompt: str,
    temperature: float,
    max_completion_tokens: int,
    is_multimodal: bool = False
) -> Generator[list, None, None]:
    if is_multimodal:
        messages = prepare_multimodal_messages(message, history, system_prompt)
    else:
        messages = prepare_chat_messages(message, history, system_prompt)

    with client as c:
        stream = c.chat.completions.create(
            model=client.model_name,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            stream=True,
        )

    yield from parse_stream(stream)
