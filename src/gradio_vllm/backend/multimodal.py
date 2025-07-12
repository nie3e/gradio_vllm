from typing import Generator

from gradio_vllm.main import image_to_base64, client, get_model_name


def inference(
    message: dict, history: list[str], system_prompt: str,
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
