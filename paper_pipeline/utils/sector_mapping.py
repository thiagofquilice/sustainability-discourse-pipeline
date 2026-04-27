"""Mappings and helpers for corporate SIC sector handling."""

from __future__ import annotations

SECTOR_MAPPING = {
    "01": "agriculture_crops",
    "02": "agriculture_livestock",
    "10": "metal_mining",
    "13": "oil_gas",
    "20": "food_manufacturing",
}

ALLOWED_SIC2 = frozenset(SECTOR_MAPPING)


def compute_sic2(sic: int | str | None) -> str | None:
    if sic is None or sic == "":
        return None
    try:
        value = int(sic)
    except (TypeError, ValueError):
        return None
    return f"{value // 100:02d}"


def map_industry(sic2: str | None) -> str | None:
    if sic2 is None:
        return None
    return SECTOR_MAPPING.get(sic2)
