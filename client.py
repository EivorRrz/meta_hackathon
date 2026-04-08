"""
Typed convenience client for the customer-support OpenEnv environment.

Usage (async):
    >>> from client import connect, SupportAction
    >>> async with connect("http://localhost:8000") as env:
    ...     result = await env.reset(task="support_classify")
    ...     result = await env.step({"action_type": "classify", "content": "billing", "metadata": {}})
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from openenv.core.generic_client import GenericEnvClient

from support_env.models import SupportAction, SupportObservation  # noqa: F401


@asynccontextmanager
async def connect(
    base_url: str = "http://127.0.0.1:8000",
    *,
    docker_image: str | None = None,
) -> AsyncIterator[GenericEnvClient]:
    """Return a ready-to-use GenericEnvClient, from URL or Docker image."""
    if docker_image:
        env = await GenericEnvClient.from_docker_image(docker_image)
    else:
        env = GenericEnvClient(base_url=base_url)
    try:
        async with env:
            yield env
    finally:
        try:
            await env.close()
        except Exception:
            pass
