"""FastAPI application exposing the on-demand anomaly inference API."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .cache import ModelCache, ModelLease
from .config import ConfigurationError, Settings, load_settings
from .errors import (
    InferenceError,
    ModelBusyError,
    ModelCacheError,
    ModelLoadError,
    ModelTooLargeError,
    UnknownModelError,
)
from .features import FEATURE_NAMES, extract_features
from .models import Manifest, load_manifest

logger = logging.getLogger(__name__)

MAX_SAMPLES_PER_SIGNAL = 10_000
_DISCONNECT_POLL_SECONDS = 0.02


class MeasurementRequest(BaseModel):
    """One measurement window for a specific device model."""

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model_id": "pump",
                    "vibration": [2.1, -1.9, 3.4, -3.0, 2.5, -2.2],
                    "current": [12.4, 12.8, 11.9, 12.6, 12.1, 12.9],
                }
            ]
        }
    }

    model_id: str = Field(
        ...,
        description="Device model id, one of the ids returned by GET /v1/models",
        examples=["pump"],
        min_length=1,
        max_length=64,
    )
    vibration: list[float] = Field(
        ...,
        description="Vibration samples in mm/s, ordered by time",
        min_length=1,
        max_length=MAX_SAMPLES_PER_SIGNAL,
    )
    current: list[float] = Field(
        ...,
        description="Drive current samples in A, aligned with the vibration window",
        min_length=1,
        max_length=MAX_SAMPLES_PER_SIGNAL,
    )

    @field_validator("vibration", "current")
    @classmethod
    def _validate_samples(cls, values: list[float]) -> list[float]:
        for value in values:
            if not np.isfinite(value):
                raise ValueError("measurements must be finite numbers")
        return values


class InferenceResponse(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    is_anomaly: bool
    decision_threshold: float
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Extracted feature vector used by the model",
    )


class ModelInfo(BaseModel):
    model_id: str
    name: str
    device_type: str
    version: str
    weights_bytes: int


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class CacheSnapshot(BaseModel):
    budget_bytes: int
    resident_bytes: int
    reserved_bytes: int
    used_bytes: int
    free_bytes: int
    models: list[dict[str, Any]]
    counters: dict[str, int]
    last_eviction: str | None


def _error_payload(code: str, message: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "message": message}
    payload.update(extra)
    return {"error": payload}


def _cache_error_response(exc: ModelCacheError) -> JSONResponse:
    headers: dict[str, str] = {}
    if isinstance(exc, UnknownModelError):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content=_error_payload(exc.code, str(exc), model_id=exc.model_id),
        )
    if isinstance(exc, ModelBusyError):
        headers["Retry-After"] = "1"
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            headers=headers,
            content=_error_payload(
                exc.code,
                str(exc),
                model_id=exc.model_id,
                budget_bytes=exc.budget_bytes,
                used_bytes=exc.used_bytes,
                required_bytes=exc.required_bytes,
            ),
        )
    if isinstance(exc, ModelTooLargeError):
        return JSONResponse(
            status_code=status.HTTP_507_INSUFFICIENT_STORAGE,
            content=_error_payload(
                exc.code,
                str(exc),
                model_id=exc.model_id,
                budget_bytes=exc.budget_bytes,
                required_bytes=exc.required_bytes,
            ),
        )
    if isinstance(exc, ModelLoadError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(exc.code, str(exc), model_id=exc.model_id),
        )
    if isinstance(exc, InferenceError):
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_payload(exc.code, str(exc)),
        )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload(exc.code, str(exc)),
    )


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        if await request.is_disconnected():
            return
        await asyncio.sleep(_DISCONNECT_POLL_SECONDS)


async def _acquire_lease(
    cache: ModelCache, model_id: str, request: Request
) -> ModelLease | None:
    """Acquire a lease, giving up cleanly if the client goes away first.

    The load runs in the cache's own task.  Cancelling ``acquire_task`` only
    removes this waiter; the shared load continues for every other request.
    """

    acquire_task = asyncio.create_task(cache.acquire(model_id), name=f"acquire:{model_id}")
    disconnect_task = asyncio.create_task(
        _wait_for_disconnect(request), name=f"disconnect:{model_id}"
    )

    async def abandon() -> None:
        """Drop this waiter; release a lease if it won the race anyway."""

        if not acquire_task.done():
            acquire_task.cancel()
        try:
            lease = await acquire_task
        except (asyncio.CancelledError, ModelCacheError):
            lease = None
        if lease is not None:
            await lease.release()

    try:
        done, _ = await asyncio.wait(
            {acquire_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if acquire_task in done:
            # Surface cache errors even if the client also disconnected.
            return acquire_task.result()

        await abandon()
        logger.info("client disconnected while waiting for model %s", model_id)
        return None
    except asyncio.CancelledError:
        # Server-side cancellation (shutdown, middleware) rather than a client
        # disconnect: same cleanup, then propagate.
        await abandon()
        raise
    finally:
        disconnect_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await disconnect_task


def _compute(lease: ModelLease, features: np.ndarray, delay_seconds: float) -> tuple[float, bool]:
    """Synchronous scoring body.

    The simulated latency lives inside this function (rather than around it)
    so that "computation is still running" is an explicit, testable window.
    """

    if delay_seconds > 0:
        # Simulated device-side model latency.  Blocking here would stall the
        # event loop, so the API runs this whole function in a worker thread.
        time.sleep(delay_seconds)
    model = lease.model
    score = model.score(features)
    return score, model.is_anomalous(score)


async def _execute_inference(
    lease: ModelLease, features: np.ndarray, delay_seconds: float
) -> tuple[float, bool]:
    """Run the computation and release the lease only when it is truly over.

    The release lives in this coroutine's ``finally`` rather than in the request
    handler: a request may be cancelled (client disconnect, server shutdown)
    while the worker thread is still scoring, and the cache must keep treating
    the model as in use until that work ends.
    """

    try:
        return await asyncio.to_thread(_compute, lease, features, delay_seconds)
    finally:
        await lease.release()


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    # A cancelled request detaches from its computation.  Retrieve any error so
    # asyncio does not warn about an unobserved task exception.
    if not task.cancelled():
        task.exception()


async def _run_inference(
    lease: ModelLease,
    features: np.ndarray,
    delay_seconds: float,
    pending: set[asyncio.Task[Any]],
) -> tuple[float, bool]:
    """Await scoring without letting a cancellation abort or shorten it.

    ``shield`` protects the computation task from the request's cancellation;
    the task owns the lease and releases it once the thread finishes.  The task
    is also registered in ``pending`` so the app's lifespan can wait for it.
    """

    task = asyncio.create_task(
        _execute_inference(lease, features, delay_seconds),
        name="inference",
    )
    pending.add(task)

    def _done(finished: asyncio.Task[Any]) -> None:
        pending.discard(finished)
        _consume_task_result(finished)

    task.add_done_callback(_done)
    # Awaiting a shield means a cancellation of the request detaches from the
    # computation instead of cancelling it; the task releases the lease itself.
    return await asyncio.shield(task)


def create_app(
    settings: Settings | None = None,
    *,
    manifest: Manifest | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings or load_settings()
        active_manifest = manifest or load_manifest(active_settings.model_dir)
        cache = ModelCache(
            active_manifest,
            active_settings.model_budget_bytes,
            active_settings.model_load_delay_seconds,
        )
        app.state.settings = active_settings
        app.state.cache = cache
        app.state.feature_names = active_manifest.feature_names
        # Detached computations keep running after their request is gone; the
        # lifespan waits for them so no lease outlives this app instance.
        pending: set[asyncio.Task[Any]] = set()
        app.state.pending_inferences = pending
        logger.info(
            "inference service ready: %d models, budget=%d bytes, inference_delay=%dms",
            len(active_manifest.descriptors),
            active_settings.model_budget_bytes,
            active_settings.inference_delay_ms,
        )
        try:
            yield
        finally:
            if pending:
                await asyncio.gather(*list(pending), return_exceptions=True)
            await cache.aclose()

    app = FastAPI(
        title="Device Anomaly Inference API",
        description=(
            "按设备型号提交振动与电流测值，按需装载固定权重逻辑回归模型并返回异常分数。\n\n"
            "模型权重按需驻留，驻留与装载预留合计不超预算；空间不足且无空闲模型时返回 503。"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def get_cache(request: Request) -> ModelCache:
        return request.app.state.cache

    @app.get("/healthz", summary="Liveness probe", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get(
        "/v1/models",
        response_model=ModelListResponse,
        summary="List available device models",
        tags=["inference"],
    )
    async def list_models(cache: ModelCache = Depends(get_cache)) -> ModelListResponse:
        return ModelListResponse(models=[ModelInfo(**item) for item in cache.list_models()])

    @app.post(
        "/v1/infer",
        response_model=InferenceResponse,
        summary="Score one measurement window",
        responses={
            404: {"description": "Unknown device model"},
            422: {"description": "Invalid measurement payload"},
            500: {"description": "Model load or inference failure"},
            503: {"description": "Model cache busy, retry later"},
            507: {"description": "Model cannot fit into the configured budget"},
        },
        tags=["inference"],
    )
    async def infer(
        payload: MeasurementRequest,
        request: Request,
        cache: ModelCache = Depends(get_cache),
    ) -> InferenceResponse | JSONResponse:
        try:
            features = extract_features(payload.vibration, payload.current)
        except ValueError as exc:
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=_error_payload("invalid_measurement", str(exc)),
            )

        try:
            lease = await _acquire_lease(cache, payload.model_id, request)
        except ModelCacheError as exc:
            return _cache_error_response(exc)

        if lease is None:
            # Client is gone; nothing to return.  A 499-style status is used
            # because the response will not be delivered anyway.
            return JSONResponse(
                status_code=499,
                content=_error_payload(
                    "client_disconnected",
                    "client disconnected while waiting for the model",
                ),
            )

        delay = request.app.state.settings.inference_delay_seconds
        try:
            # Lease release is owned by _execute_inference: doing it here would
            # release the model early when the request is cancelled mid-compute.
            score, is_anomaly = await _run_inference(
                lease, features, delay, request.app.state.pending_inferences
            )
            model = lease.model
        except ModelCacheError as exc:
            return _cache_error_response(exc)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("inference failed for model %s", payload.model_id)
            return _cache_error_response(InferenceError(f"inference failed: {exc}"))

        return InferenceResponse(
            model_id=model.model_id,
            model_name=model.name,
            model_version=model.version,
            anomaly_score=score,
            is_anomaly=is_anomaly,
            decision_threshold=model.decision_threshold,
            features={
                name: float(value)
                for name, value in zip(FEATURE_NAMES, features, strict=True)
            },
        )

    @app.get(
        "/v1/cache",
        response_model=CacheSnapshot,
        summary="Inspect model cache usage",
        tags=["ops"],
    )
    async def cache_snapshot(cache: ModelCache = Depends(get_cache)) -> CacheSnapshot:
        return CacheSnapshot(**cache.snapshot())

    app.state.settings = None
    app.state.cache = None
    return app


try:
    app = create_app()
except ConfigurationError:
    logger.exception("invalid configuration")
    raise
