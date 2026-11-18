"""Domain errors surfaced by the inference cache."""

from __future__ import annotations


class ModelCacheError(RuntimeError):
    """Base class for cache-related failures."""

    code = "model_cache_error"


class UnknownModelError(ModelCacheError):
    code = "unknown_model"

    def __init__(self, model_id: str) -> None:
        super().__init__(f"unknown model: {model_id}")
        self.model_id = model_id


class ModelBusyError(ModelCacheError):
    """No budget available and every resident model is in use."""

    code = "model_busy"

    def __init__(
        self,
        model_id: str,
        *,
        budget_bytes: int,
        used_bytes: int,
        required_bytes: int,
    ) -> None:
        super().__init__(
            "model cache is busy: no idle model can be evicted to make room"
        )
        self.model_id = model_id
        self.budget_bytes = budget_bytes
        self.used_bytes = used_bytes
        self.required_bytes = required_bytes


class ModelTooLargeError(ModelCacheError):
    """The model's weights can never fit into the configured budget."""

    code = "model_too_large"

    def __init__(self, model_id: str, *, budget_bytes: int, required_bytes: int) -> None:
        super().__init__(f"model {model_id} needs {required_bytes} bytes but the budget is {budget_bytes}")
        self.model_id = model_id
        self.budget_bytes = budget_bytes
        self.required_bytes = required_bytes


class ModelLoadError(ModelCacheError):
    """Reading or validating a model file failed."""

    code = "model_load_failed"

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"failed to load model {model_id}: {reason}")
        self.model_id = model_id
        self.reason = reason


class InferenceError(ModelCacheError):
    code = "inference_failed"
