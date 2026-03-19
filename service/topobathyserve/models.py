from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator  # type: ignore


class LatLonRequest(BaseModel):
    lat: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees")
    lon: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees")


class BoundingBoxRequest(BaseModel):
    north: float = Field(..., ge=-90, le=90)
    south: float = Field(..., ge=-90, le=90)
    west: float = Field(..., ge=-180, le=180)
    east: float = Field(..., ge=-180, le=180)

    @field_validator("north")
    def north_must_be_greater_than_south(cls, v: float, values: ValidationInfo) -> float:  # noqa: N805
        if "south" in values.data and v <= values.data["south"]:
            raise ValueError("north must be greater than south")
        return v


class HydrateRequest(BaseModel):
    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="Bounding box as (west, south, east, north) in WGS84 decimal degrees",
        examples=[[-73.82607, 40.75870, -72.85789, 41.20501]],
    )
    resolution: float = Field(30.0, description="Output resolution in meters", examples=[30.0])


class ElevationResponse(BaseModel):
    elevation: float | None
    unit: str = "meters"


class TIDReportResponse(BaseModel):
    report: dict[str, Any]
    description: str = "Percentage of data sources by quality tier."


class TileMetadataResponse(BaseModel):
    z: int
    x: int
    y: int
    bounds: dict[str, float]
    created_at: str | None = None
    fusion_sources: str | None = None
    cache_status: str
    request_params: str | None = None
    provenance_dict: dict[str, dict[str, Any]] | None = None
