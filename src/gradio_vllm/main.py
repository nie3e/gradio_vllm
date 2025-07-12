import os

from openai import OpenAI

from gradio_vllm.frontend import interface

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
BASE_URL = "http://series2.rav:8000/v1"

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")


def get_model_name() -> str:
    model_name = client.models.list().data[0].id
    return model_name


def main() -> int:
    """Entry point function for the web app."""
    demo = interface.create_app()
    demo.queue(max_size=10).launch()

    return 0


if __name__ == "__main__":
    exit(main())
