"""Funciones de análisis para la simulación."""
from __future__ import annotations

from typing import Dict
import numpy as np

from .ca_model import LAND_RURAL, LAND_URBAN, LAND_VEG, LAND_WATER


def compute_metrics(T: np.ndarray, land_use: np.ndarray) -> Dict[str, float]:
    """Calcula métricas simples en un instante.

    Retorna promedios por tipo de suelo y la intensidad UHI.
    """
    mask_urban = land_use == LAND_URBAN
    mask_rural = land_use == LAND_RURAL
    mask_veg = land_use == LAND_VEG
    mask_water = land_use == LAND_WATER

    metrics = {
        "mean_all": float(T.mean()),
        "mean_urban": float(T[mask_urban].mean()) if mask_urban.any() else np.nan,
        "mean_rural": float(T[mask_rural].mean()) if mask_rural.any() else np.nan,
        "mean_veg": float(T[mask_veg].mean()) if mask_veg.any() else np.nan,
        "mean_water": float(T[mask_water].mean()) if mask_water.any() else np.nan,
    }

    nonurban_mask = mask_rural | mask_veg | mask_water
    if mask_urban.any() and nonurban_mask.any():
        metrics["uhi"] = float(T[mask_urban].mean() - T[nonurban_mask].mean())
    else:
        metrics["uhi"] = np.nan

    return metrics
