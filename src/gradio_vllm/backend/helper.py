from typing import Iterable, Generator
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
