import os
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")


class VLLMClient:
    def __init__(self) -> None:
        self._client: OpenAI | None = None
        self._base_url: str
        self._api_key: str
        self._model_name: str | None = None

    def set_connection(self, base_url: str, api_key: str = "k") -> str | None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        try:
            self._model_name = self._client.models.list().data[0].id
        except Exception:
            self._client = None
            return "No connection"
        return self._model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def __enter__(self) -> OpenAI:
        if not self._model_name or not self._client:
            raise RuntimeError("No VLLM connection")
        return self._client

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


client = VLLMClient()
