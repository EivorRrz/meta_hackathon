"""Server entry point for openenv validate compatibility."""

from support_env.server.app import app  # noqa: F401


def main() -> None:
    import uvicorn

    uvicorn.run("support_env.server.app:app", host="0.0.0.0", port=8000, factory=False)


if __name__ == "__main__":
    main()
