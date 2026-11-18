import asyncio

import pytest

from app.cache import EntryState, ModelCache
from app.errors import ModelBusyError, ModelLoadError, ModelTooLargeError, UnknownModelError
from app.models import Manifest
from tests.conftest import MODEL_BYTES


def measurement(model_id: str) -> dict:
    return {"model_id": model_id, "vibration": [1.0, 2.0, 1.0], "current": [5.0, 6.0, 5.0]}


async def test_cold_requests_for_same_model_load_only_once(settings, manifest: Manifest):
    settings = settings.__class__(**{**settings.__dict__, "model_load_delay_ms": 120})
    cache = ModelCache(manifest, settings.model_budget_bytes, settings.model_load_delay_seconds)

    leases = await asyncio.gather(*(cache.acquire("alpha") for _ in range(8)))
    try:
        snapshot = cache.snapshot()
        assert snapshot["counters"]["load_attempts"] == 1
        assert snapshot["counters"]["load_succeeded"] == 1
        assert snapshot["counters"]["cold_requests"] == 1
        assert snapshot["resident_bytes"] == MODEL_BYTES
        assert len(leases) == 8
        assert all(lease.model.model_id == "alpha" for lease in leases)
    finally:
        for lease in leases:
            await lease.release()
    await cache.aclose()


async def test_reservations_count_towards_budget_while_loading(settings, manifest: Manifest):
    # Budget holds exactly two models.
    cache = ModelCache(manifest, MODEL_BYTES * 2, load_delay_seconds=0.15)

    first = asyncio.create_task(cache.acquire("alpha"))
    second = asyncio.create_task(cache.acquire("beta"))
    await asyncio.sleep(0.03)

    snapshot = cache.snapshot()
    assert snapshot["reserved_bytes"] == MODEL_BYTES * 2
    assert snapshot["resident_bytes"] == 0
    assert snapshot["free_bytes"] == 0

    third = asyncio.create_task(cache.acquire("gamma"))
    try:
        with pytest.raises(ModelBusyError):
            await third
        assert cache.snapshot()["counters"]["busy_rejections"] == 1

        leases = await asyncio.gather(first, second)
        snapshot = cache.snapshot()
        assert snapshot["resident_bytes"] == MODEL_BYTES * 2
        assert snapshot["reserved_bytes"] == 0
        assert snapshot["used_bytes"] <= snapshot["budget_bytes"]
        for lease in leases:
            await lease.release()
    finally:
        for task in (first, second, third):
            task.cancel()
        await asyncio.gather(first, second, third, return_exceptions=True)
        await cache.aclose()


async def test_eviction_only_targets_idle_models(settings, manifest: Manifest):
    cache = ModelCache(manifest, MODEL_BYTES * 2)

    alpha = await cache.acquire("alpha")
    beta = await cache.acquire("beta")
    assert cache.snapshot()["resident_bytes"] == MODEL_BYTES * 2

    # gamma needs room, alpha and beta are both in flight -> busy.
    with pytest.raises(ModelBusyError):
        await cache.acquire("gamma")
    assert cache.snapshot()["counters"]["evictions"] == 0

    await beta.release()
    gamma = await cache.acquire("gamma")
    snapshot = cache.snapshot()
    assert snapshot["counters"]["evictions"] == 1
    assert snapshot["last_eviction"] == "beta"
    assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "gamma"}
    assert snapshot["used_bytes"] <= snapshot["budget_bytes"]
    assert alpha.model.model_id == "alpha"

    await alpha.release()
    await gamma.release()
    await cache.aclose()


async def test_lru_eviction_prefers_least_recently_used(settings, manifest: Manifest):
    """An idle model is evicted in LRU order to make room for a cold model."""

    cache = ModelCache(manifest, MODEL_BYTES * 2)
    alpha = await cache.acquire("alpha")
    await alpha.release()

    lease = await cache.acquire("beta")
    await lease.release()

    # Evicts the least recently used (alpha) without error.
    gamma = await cache.acquire("gamma")
    assert cache.snapshot()["last_eviction"] == "alpha"
    await gamma.release()
    await cache.aclose()


async def test_busy_rejection_does_not_disturb_resident_models(settings, manifest: Manifest):
    cache = ModelCache(manifest, MODEL_BYTES * 2)
    alpha = await cache.acquire("alpha")
    beta = await cache.acquire("beta")

    with pytest.raises(ModelBusyError):
        await cache.acquire("gamma")

    snapshot = cache.snapshot()
    assert snapshot["resident_bytes"] == MODEL_BYTES * 2
    assert {entry["model_id"] for entry in snapshot["models"]} == {"alpha", "beta"}

    await alpha.release()
    await beta.release()
    await cache.aclose()


async def test_oversized_model_is_rejected_with_507_style_error(settings, manifest: Manifest):
    cache = ModelCache(manifest, MODEL_BYTES - 1)
    with pytest.raises(ModelTooLargeError):
        await cache.acquire("alpha")
    assert cache.snapshot()["used_bytes"] == 0
    await cache.aclose()


async def test_unknown_model(settings, manifest: Manifest):
    cache = ModelCache(manifest, MODEL_BYTES * 2)
    with pytest.raises(UnknownModelError):
        await cache.acquire("nope")
    await cache.aclose()


async def test_failed_load_clears_reservation_and_other_models_keep_working(
    settings, manifest: Manifest, model_dir
):
    cache = ModelCache(manifest, MODEL_BYTES * 2, load_delay_seconds=0.05)
    (model_dir / "alpha.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(ModelLoadError):
        await cache.acquire("alpha")

    snapshot = cache.snapshot()
    assert snapshot["reserved_bytes"] == 0
    assert snapshot["resident_bytes"] == 0
    assert snapshot["counters"]["load_failed"] == 1

    lease = await cache.acquire("beta")
    assert lease.model.model_id == "beta"
    await lease.release()

    # A later request retries the broken model once the file is repaired.
    from tests.conftest import write_model

    write_model(model_dir, "alpha")
    retried = await cache.acquire("alpha")
    snapshot = cache.snapshot()
    assert snapshot["counters"]["load_failed"] == 1
    assert snapshot["counters"]["load_succeeded"] == 2
    await retried.release()
    await cache.aclose()


async def test_failed_shared_load_wakes_every_waiter(settings, manifest: Manifest, model_dir):
    cache = ModelCache(manifest, MODEL_BYTES * 2, load_delay_seconds=0.05)
    (model_dir / "alpha.json").write_text("garbage", encoding="utf-8")

    results = await asyncio.gather(
        *(cache.acquire("alpha") for _ in range(5)), return_exceptions=True
    )
    assert all(isinstance(result, ModelLoadError) for result in results)
    assert cache.snapshot()["counters"]["load_attempts"] == 1
    assert cache.snapshot()["reserved_bytes"] == 0
    await cache.aclose()


async def test_whole_resident_set_stays_within_budget(settings, manifest: Manifest):
    cache = ModelCache(manifest, MODEL_BYTES * 2, load_delay_seconds=0.01)

    for _ in range(3):
        leases = await asyncio.gather(
            cache.acquire("alpha"), cache.acquire("beta"), cache.acquire("gamma"),
            return_exceptions=True,
        )
        snapshot = cache.snapshot()
        assert snapshot["used_bytes"] <= snapshot["budget_bytes"]
        assert snapshot["resident_bytes"] + snapshot["reserved_bytes"] <= snapshot["budget_bytes"]
        for lease in leases:
            if not isinstance(lease, Exception):
                await lease.release()

    await cache.aclose()


async def test_entry_state_transitions(settings, manifest: Manifest):
    cache = ModelCache(manifest, MODEL_BYTES * 2)
    lease = await cache.acquire("alpha")
    entries = {entry["model_id"]: entry for entry in cache.snapshot()["models"]}
    assert entries["alpha"]["state"] == EntryState.READY.value
    assert entries["alpha"]["inflight"] == 1
    await lease.release()
    entries = {entry["model_id"]: entry for entry in cache.snapshot()["models"]}
    assert entries["alpha"]["inflight"] == 0
    await cache.aclose()
