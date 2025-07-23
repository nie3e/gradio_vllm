from typing import Generator

from gradio_vllm.backend.helper import (parse_stream,
                                        prepare_multimodal_messages)
from gradio_vllm.backend.vllm_client import client


def inference(
    message: dict,
    history: list[dict],
    system_prompt: str,
    temperature: float,
    max_completion_tokens: int
) -> Generator[list, None, None]:
    messages = prepare_multimodal_messages(message, history, system_prompt)

    with client as c:
        stream = c.chat.completions.create(  # noqa
            model=client.model_name,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            stream=True,
        )

    yield from parse_stream(stream)
