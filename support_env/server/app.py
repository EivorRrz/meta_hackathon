from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from support_env.models import SupportAction, SupportObservation
from support_env.server.environment import CustomerSupportEnvironment

app = create_app(
    CustomerSupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="customer_support_env",
)


def main() -> None:
    import uvicorn

    uvicorn.run("support_env.server.app:app", host="0.0.0.0", port=8000, factory=False)


if __name__ == "__main__":
    main()
