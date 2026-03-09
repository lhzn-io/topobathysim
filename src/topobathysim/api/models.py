from typing import Any

from pydantic import BaseModel


class CacheTierInfo(BaseModel):
    number: int
    name: str
    short_desc: str
    long_desc: str
    purge_reason: str
    warning: bool


class CacheTierSummary(BaseModel):
    number: int
    name: str
    items: int
    bytes: int
    mb: float
    warning: bool


class CacheSummary(BaseModel):
    cache_root: str
    total_bytes: int
    total_mb: float
    tiers: list[CacheTierSummary]
    last_updated: float


class TileSubdir(BaseModel):
    label: str
    files: int
    bytes: int


class Tier1Detail(BaseModel):
    tile_subdirs: list[TileSubdir]
    meta_count: int
    npz_count: int


class ZoomDetail(BaseModel):
    count: int
    bytes: int
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


class Tier2Detail(BaseModel):
    count: int
    bytes: int
    newest: float
    oldest: float
    by_zoom: dict[int, ZoomDetail]


class ProviderDetail(BaseModel):
    name: str
    count: int
    bytes: int
    newest: float
    oldest: float


class Tier3Detail(BaseModel):
    providers: list[ProviderDetail]


class DiscoveryDetail(BaseModel):
    label: str
    ok: bool
    bytes: int
    entries: int
    mtime: float


class Tier4Detail(BaseModel):
    discovery: list[DiscoveryDetail]
    bluetopo_scheme: dict[str, Any]
    bluetopo_sidecars: dict[str, Any]
    topobathy_zips: dict[str, Any]


class RawFileDetail(BaseModel):
    provider: str
    count: int
    bytes: int
    newest: float
    oldest: float


class Tier5Detail(BaseModel):
    raw_files: list[RawFileDetail]


class CacheDetail(BaseModel):
    cache_root: str
    total_bytes: int
    tier_1: Tier1Detail
    tier_2: Tier2Detail
    tier_3: Tier3Detail
    tier_4: Tier4Detail
    tier_5: Tier5Detail


class PurgeRequest(BaseModel):
    tiers: list[int]
    dry_run: bool = True
    yes: bool = False


class PurgeResult(BaseModel):
    status: str
    mode: str
    message: str | None = None
    summary: dict[str, dict[str, int]]
    paths: list[str]


class CheckRequest(BaseModel):
    clean: bool = False
    lock_timeout: int = 3600


class CheckResult(BaseModel):
    status: str
    checks: dict[str, Any]
    orphans: dict[str, Any]


class OrphansRequest(BaseModel):
    clean: bool = False


class OrphansResult(BaseModel):
    status: str
    mode: str
    orphans: dict[str, Any]
