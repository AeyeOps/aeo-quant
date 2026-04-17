"""Pure-compute workload functions used by examples and the harness daemon.

Each workload exposes ``run(model, tokenizer, **kwargs) -> dict`` where the
returned dict is JSON-serializable. No filesystem I/O, no baseline comparison,
no pretty printing — those live in the CLI wrappers so the server stays a
thin queue around the GPU work.
"""

from . import parity

# Registry used by the harness server to dispatch by workload name.
WORKLOADS = {
    "parity": parity.run,
}

__all__ = ["WORKLOADS", "parity"]
