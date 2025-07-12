from gradio_vllm.frontend import interface


def main() -> int:
    """Entry point function for the web app."""
    demo = interface.create_app()
    demo.queue(max_size=10).launch()

    return 0


if __name__ == "__main__":
    exit(main())
