from pydantic import BaseModel, Field, ValidationInfo, field_validator


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


class ElevationResponse(BaseModel):
    elevation: float
    unit: str = "meters"


class TIDReportResponse(BaseModel):
    report: dict[str, float]
    description: str = "Percentage of data sources by quality tier."
