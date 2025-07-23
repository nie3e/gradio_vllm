import os
import pytest

from gradio_vllm.backend.helper import (
    prepare_chat_messages, prepare_multimodal_messages, image_to_base64
)

dir_path = os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "message, history, system_prompt, expected_result",
    [
        ("Test", [], "", [{"role": "user", "content": "Test"}]),
        (
            "Test",
            [],
            "system_msg",
            [
                {"role": "system", "content": "system_msg"},
                {"role": "user", "content": "Test"},
            ],
        ),
        (
            "Test",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
            "",
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "Test"}
            ],
        ),
    ],
)
def test_prepare_chat_messages(
        message, history, system_prompt, expected_result
):
    result = prepare_chat_messages(message, history, system_prompt)
    assert result == expected_result


@pytest.mark.parametrize(
    "message, history, system_prompt, expected_result",
    [
        (
            {"files": [], "text": "Test"},
            [],
            "",
            [
                {"role": "user", "content": [{"text": "Test", "type": "text"}]}
            ]
        ),
        (
            {"files": [], "text": "Test"},
            [],
            "system_msg",
            [
                {"role": "system", "content": "system_msg"},
                {"role": "user", "content": [{"text": "Test", "type": "text"}]}
            ]
        ),
        (
            {
                "files": [f"{dir_path}/resources/test_image.png"],
                "text": "Test"
            },
            [],
            "system_msg",
            [
                {"role": "system", "content": "system_msg"},
                {"role": "user",
                 "content": [
                     {"text": "Test", "type": "text"},
                     {"image_url":
                         {"url": image_to_base64(
                            f"{dir_path}/resources/test_image.png"
                         )},
                         "type": "image_url"}
                    ]
                 }
            ]
        )
    ]
)
def test_prepare_multimodal_messages(
        message, history, system_prompt, expected_result
):
    result = prepare_multimodal_messages(message, history, system_prompt)
    assert result == expected_result
