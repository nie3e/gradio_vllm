import os

from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def get_model_name() -> str:
    model_name = client.models.list().data[0].id
    return model_name
