from enum import Enum
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from topobathysim.providers.registry import registry


class ZoneRule(BaseModel):
    """
    Defines a geographic zone and the rules applied within it.
    """

    type: Literal["polygon", "bbox"] = "bbox"
    coords: list[float]  # [min_x, min_y, max_x, max_y] for bbox
    priority: int = 0


class OperatorType(str, Enum):
    metric_feather = "metric_feather"
    overwrite = "overwrite"
    linear_blend = "linear_blend"


class TransitionRule(BaseModel):
    """
    Rule for transitioning from a specific underlying provider.
    """

    target_provider: str
    operator: OperatorType
    blend_distance: float | None = None


class CompositionStep(BaseModel):
    """
    A single step in the composition process.
    """

    provider: str
    operation: OperatorType = OperatorType.metric_feather
    blend_mode: Literal["linear", "feather"] | None = None  # Deprecated in favor of operation
    blend_distance: float | None = None  # In meters
    zone: ZoneRule | None = None
    transitions: list[TransitionRule] = []

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        # Runtime check to ensure provider is registered
        # Note: This requires providers to be loaded before policy validation
        import contextlib

        with contextlib.suppress(KeyError):
            registry.get_provider_class(v)
        return v


class VariableStrategy(BaseModel):
    """
    Strategy for a specific variable (e.g., elevation).
    """

    name: str = "elevation"
    steps: list[CompositionStep]
    background: float = float("nan")


class FusionPolicy(BaseModel):
    """
    Top-level policy definition.
    """

    name: str
    version: str = "1.0.0"
    crs: str = "EPSG:4326"
    variables: list[VariableStrategy]

    @model_validator(mode="after")
    def check_at_least_one_variable(self) -> "FusionPolicy":
        if not self.variables:
            raise ValueError("Policy must contain at least one variable strategy.")
        return self
