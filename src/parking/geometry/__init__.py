"""Spot geometry (§2 stages 2-3): polygons per spot, crop + perspective warp."""

from parking.geometry.spots import (
    LotConfig,
    RoiExtractor,
    Spot,
    load_lot,
    order_corners,
    save_lot,
)

__all__ = [
    "LotConfig",
    "RoiExtractor",
    "Spot",
    "load_lot",
    "order_corners",
    "save_lot",
]
