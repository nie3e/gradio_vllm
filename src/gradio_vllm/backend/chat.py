from typing import Generator

from gradio_vllm.backend.helper import parse_stream
from gradio_vllm.backend.vllm_client import client


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

    with client as c:
        stream = c.chat.completions.create(
            model=client.model_name,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            stream=True,
        )

    yield from parse_stream(stream)
