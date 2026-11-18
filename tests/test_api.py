import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import create_app
from app.cache import ModelCache
from app.models import Manifest
from tests.conftest import MODEL_BYTES


def payload(model_id: str = "alpha", samples: int = 8) -> dict:
    return {
        "model_id": model_id,
        "vibration": [1.0 + (index % 3) for index in range(samples)],
        "current": [5.0 + (index % 2) for index in range(samples)],
    }


async def test_health_models_and_swagger(client: AsyncClient):
    health = await client.get("/healthz")
    assert health.status_code == 200

    models = await client.get("/v1/models")
    assert models.status_code == 200
    assert [item["model_id"] for item in models.json()["models"]] == ["alpha", "beta", "gamma"]

    openapi = await client.get("/openapi.json")
    assert openapi.status_code == 200
    assert "/v1/infer" in openapi.json()["paths"]


async def test_inference_is_deterministic(client: AsyncClient):
    first = await client.post("/v1/infer", json=payload())
    second = await client.post("/v1/infer", json=payload())
    assert first.status_code == 200
    assert first.json() == second.json()
    body = first.json()
    assert 0.0 <= body["anomaly_score"] <= 1.0
    assert body["model_id"] == "alpha"
    assert body["model_version"] == "1.0.0"
    assert set(body["features"]) == {
        "vibration_mean", "vibration_rms", "vibration_peak", "vibration_std",
        "vibration_crest", "vibration_kurtosis", "current_mean", "current_rms",
        "current_peak", "current_std", "current_crest", "current_kurtosis",
    }


async def test_unknown_model_returns_404(client: AsyncClient):
    response = await client.post("/v1/infer", json=payload("nope"))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "unknown_model"


async def test_invalid_samples_return_422(client: AsyncClient):
    response = await client.post("/v1/infer", json={"model_id": "alpha", "vibration": [], "current": [1.0]})
    assert response.status_code == 422


async def test_mismatched_model_size_returns_507(manifest: Manifest, model_dir, settings):
    small = settings.__class__(**{**settings.__dict__, "model_budget_bytes": MODEL_BYTES - 1})
    app = create_app(small, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/v1/infer", json=payload())
            assert response.status_code == 507
            assert response.json()["error"]["code"] == "model_too_large"


async def test_busy_cache_returns_503_with_retry_after(manifest: Manifest, model_dir, settings):
    busy_settings = settings.__class__(
        **{**settings.__dict__, "model_budget_bytes": MODEL_BYTES, "inference_delay_ms": 300}
    )
    app = create_app(busy_settings, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/infer", json=payload("alpha")))
            await asyncio.sleep(0.1)
            second = await client.post("/v1/infer", json=payload("beta"))
            assert second.status_code == 503
            assert second.headers["retry-after"] == "1"
            assert second.json()["error"]["code"] == "model_busy"
            assert (await first).status_code == 200

            snapshot = await client.get("/v1/cache")
            assert snapshot.json()["used_bytes"] <= busy_settings.model_budget_bytes


async def test_load_failure_returns_500_and_cache_stays_usable(manifest: Manifest, model_dir, settings):
    app = create_app(settings, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            (model_dir / "alpha.json").write_text("broken", encoding="utf-8")
            failed = await client.post("/v1/infer", json=payload("alpha"))
            assert failed.status_code == 500
            assert failed.json()["error"]["code"] == "model_load_failed"

            healthy = await client.post("/v1/infer", json=payload("beta"))
            assert healthy.status_code == 200

            snapshot = (await client.get("/v1/cache")).json()
            assert snapshot["reserved_bytes"] == 0
            assert snapshot["resident_bytes"] == MODEL_BYTES


async def test_concurrent_cold_requests_load_once(manifest: Manifest, model_dir, settings):
    slow = settings.__class__(
        **{**settings.__dict__, "model_load_delay_ms": 150, "inference_delay_ms": 10}
    )
    app = create_app(slow, manifest=manifest)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            responses = await asyncio.gather(
                *(client.post("/v1/infer", json=payload("alpha")) for _ in range(10))
            )
            assert {response.status_code for response in responses} == {200}
            assert len({json.dumps(r.json()["anomaly_score"]) for r in responses}) == 1
            snapshot = (await client.get("/v1/cache")).json()
            assert snapshot["counters"]["load_attempts"] == 1
            assert snapshot["counters"]["load_succeeded"] == 1
            assert snapshot["counters"]["cold_requests"] == 1


async def test_disconnected_waiter_does_not_cancel_shared_load(
    manifest: Manifest, model_dir, settings
):
    """The first waiter drops while the shared load is still running."""

    slow = settings.__class__(
        **{**settings.__dict__, "model_load_delay_ms": 250}
    )
    app = create_app(slow, manifest=manifest)
    async with app.router.lifespan_context(app):
        cache: ModelCache = app.state.cache
        disconnect_task = asyncio.create_task(cache.acquire("alpha"))
        await asyncio.sleep(0.02)
        disconnect_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await disconnect_task

        # Still loading; a second request joins the same in-flight load.
        snapshot = cache.snapshot()
        assert snapshot["reserved_bytes"] == MODEL_BYTES
        assert snapshot["counters"]["load_attempts"] == 1

        lease = await cache.acquire("alpha")
        try:
            assert lease.model.model_id == "alpha"
            final = cache.snapshot()
            assert final["counters"]["load_attempts"] == 1
            assert final["counters"]["load_succeeded"] == 1
            assert final["resident_bytes"] == MODEL_BYTES
        finally:
            await lease.release()


async def test_inference_lease_survives_request_cancellation(manifest: Manifest, model_dir, settings):
    """A cancelled inference keeps the model occupied until the work finishes."""

    app = create_app(settings, manifest=manifest)
    async with app.router.lifespan_context(app):
        cache: ModelCache = app.state.cache
        lease = await cache.acquire("alpha")
        assert cache.snapshot()["models"][0]["inflight"] == 1

        from app.api import _run_inference

        task = asyncio.create_task(
            _run_inference(
                lease,
                __import__("numpy").zeros(12),
                0.3,
                app.state.pending_inferences,
            )
        )
        await asyncio.sleep(0.05)
        started = asyncio.get_running_loop().time()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cancelled_after = asyncio.get_running_loop().time() - started

        # The request is unblocked immediately, but the detached computation
        # still owns the model.
        assert cancelled_after < 0.1
        assert cache.snapshot()["models"][0]["inflight"] == 1

        # The lease is dropped only once the simulated computation has ended.
        while cache.snapshot()["models"][0]["inflight"] > 0:  # noqa: ASYNC110
            await asyncio.sleep(0.01)
        elapsed = asyncio.get_running_loop().time() - started
        assert elapsed >= 0.2
