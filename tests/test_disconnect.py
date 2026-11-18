"""End-to-end concurrency semantics through the real ASGI app.

These tests inject ``http.disconnect`` at the ASGI layer, which is the closest
equivalent to a dropped client connection available without a live socket.
"""

from __future__ import annotations

import asyncio

import pytest

from app.api import create_app
from app.models import Manifest
from tests.conftest import MODEL_BYTES
from tests.helpers import RawASGIRequest


def payload(model_id: str = "alpha", samples: int = 16) -> dict:
    return {
        "model_id": model_id,
        "vibration": [1.0 + (index % 4) for index in range(samples)],
        "current": [5.0 + (index % 2) for index in range(samples)],
    }


async def test_disconnect_while_waiting_leaves_shared_load_intact(
    manifest: Manifest, settings
):
    slow = settings.__class__(**{**settings.__dict__, "model_load_delay_ms": 300})
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        dropped = RawASGIRequest(app, "/v1/infer", payload("alpha"), disconnect_after=0.05)
        dropped_task = asyncio.create_task(dropped.run())

        await asyncio.sleep(0.02)
        survivor = RawASGIRequest(app, "/v1/infer", payload("alpha"))
        survivor_task = asyncio.create_task(survivor.run())

        dropped_response = await dropped_task
        assert dropped_response.status_code == 499

        survivor_response = await survivor_task
        assert survivor_response.status_code == 200

        cache = app.state.cache
        snapshot = cache.snapshot()
        assert snapshot["counters"]["load_attempts"] == 1
        assert snapshot["counters"]["load_succeeded"] == 1
        assert snapshot["resident_bytes"] == MODEL_BYTES
        assert snapshot["reserved_bytes"] == 0


async def test_disconnect_during_inference_holds_lease_until_computation_ends(
    manifest: Manifest, settings
):
    slow = settings.__class__(**{**settings.__dict__, "inference_delay_ms": 400})
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        request = RawASGIRequest(app, "/v1/infer", payload("alpha"), disconnect_after=0.15)
        started = asyncio.get_running_loop().time()
        await request.run()
        elapsed = asyncio.get_running_loop().time() - started

        cache = app.state.cache
        snapshot = cache.snapshot()
        # By the time the ASGI call returns, the computation is over and the
        # lease has been released.
        assert snapshot["models"][0]["inflight"] == 0
        assert elapsed >= slow.inference_delay_seconds * 0.9


async def test_disconnect_during_inference_blocks_eviction_until_done(
    manifest: Manifest, settings
):
    """The model stays pinned while a disconnected request is still computing."""

    from app.errors import ModelBusyError

    slow = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES, "inference_delay_ms": 400}
    )
    app = create_app(slow, manifest=manifest)

    async with app.router.lifespan_context(app):
        request = RawASGIRequest(app, "/v1/infer", payload("alpha"), disconnect_after=0.1)
        request_task = asyncio.create_task(request.run())

        await asyncio.sleep(0.2)
        cache = app.state.cache
        snapshot = cache.snapshot()
        assert snapshot["models"][0]["inflight"] == 1

        with pytest.raises(ModelBusyError):
            await cache.acquire("beta")

        await request_task
        assert cache.snapshot()["models"][0]["inflight"] == 0

        lease = await cache.acquire("beta")
        await lease.release()
