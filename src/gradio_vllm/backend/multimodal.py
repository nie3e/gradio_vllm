from typing import Generator

from gradio_vllm.backend.helper import parse_stream, image_to_base64
from gradio_vllm.backend.vllm_client import client, get_model_name


def inference(
    message: dict,
    history: list[dict],
    system_prompt: str,
    temperature: float,
    max_completion_tokens: int
) -> Generator[list, None, None]:
    messages: list[dict] = []
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}]
    images = []
    for history_msg in [msg for msg in history if not msg.get("metadata")]:
        if type(history_msg["content"]) is tuple:
            images += history_msg["content"]
        elif history_msg["content"] and history_msg["role"] == "user":
            messages.append({
                "role": "user",
                "content":
                    [{"type": "text", "text": history_msg["content"]}] +
                    [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_base64(path)}
                        }
                        for path in images
                    ]
            })
            images = []
        elif history_msg["content"] and history_msg["role"] == "assistant":
            messages.append({"role": "assistant",
                             "content": history_msg["content"]})

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
        stream=True,
    )

    yield from parse_stream(stream)
