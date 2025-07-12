from typing import Self

import pytest

from gradio_vllm.main import parse_stream


class FakeChunk:
    class Delta:
        __slots__ = ('content',)

        def __init__(self, content: str):
            self.content = content

    class Choice:
        __slots__ = ('delta',)

        def __init__(self, content: str):
            self.delta = FakeChunk.Delta(content)

    __slots__ = ('choices',)

    def __init__(self, content: str):
        self.choices = [self.Choice(content)]


class ResponseStream:
    def __init__(self, text: str) -> None:
        self._text = text
        self._current_idx = -1
        self._len = len(text)

    def __iter__(self) -> Self:
        return self

    def __next__(self):
        self._current_idx += 1
        if self._current_idx < self._len:
            return FakeChunk(self._text[self._current_idx])
        raise StopIteration


def test_parse_stream_simple_answer():
    completion = "Test"
    stream = ResponseStream(completion)

    result = list(parse_stream(stream))[-1]

    assert result
    assert result[0].content == "Test"
    assert result[0].role == "assistant"
    assert result[0].metadata == dict()


@pytest.mark.parametrize(
    "completion",
    [
        "<think>test think</think>answer here",
        "<think>test think</think><answer>answer here</answer>"
    ]
)
def test_parse_stream_think(completion):
    stream = ResponseStream(completion)

    result = list(parse_stream(stream))[-1]

    assert result
    assert len(result) == 2

    think = result[0]
    assert think.content == "test think"
    assert think.role == "assistant"
    assert think.metadata == {"title": "⏳Thinking"}

    answer = result[1]
    assert answer.content == "answer here"
    assert answer.role == "assistant"
    assert answer.metadata == dict()
