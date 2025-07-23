import base64
from typing import Iterable, Generator, Any
from gradio import ChatMessage


def parse_stream(
    content_stream: Iterable,
) -> Generator[list[ChatMessage], None, None]:
    partial_message = ""
    think_mode = False
    answer_started = False
    messages_chat = []

    for chunk in content_stream:
        if not chunk.choices[0].delta.content:
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


def image_to_base64(image_path: str) -> str:
    with open(image_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def prepare_chat_messages(
    message: str,
    history: list[dict],
    system_prompt: str
) -> list[dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages += [
        {"role": msg["role"], "content": msg["content"]}
        for msg in history
        if not msg.get("metadata")
    ]
    messages.append({"role": "user", "content": message})

    return messages


def prepare_multimodal_messages(
    message: dict[str, Any],
    history: list[dict[str, Any]],
    system_prompt: str
) -> list[dict[str, Any]]:
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
                            "image_url": {"url": image_to_base64(path)},
                        }
                        for path in images
                    ],
            })
            images = []
        elif history_msg["content"] and history_msg["role"] == "assistant":
            messages.append({"role": "assistant",
                             "content": history_msg["content"]})

    messages.append(
        {
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
                ],
        }
    )

    return messages