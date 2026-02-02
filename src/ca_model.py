"""Modelo de autómata celular para isla de calor urbano (UHI).

El modelo usa una malla 2D con tipos de uso del suelo y una temperatura continua
por celda. En cada paso de tiempo se aplica:
- Difusión hacia vecinos (suavizado espacial).
- Forzamiento local (calor urbano, enfriamiento por vegetación/agua).
- Ciclo diurno de radiación (senoidal).
- Ruido gaussiano para variabilidad.

Las condiciones de borde son toroidales (wrap-around) para mantener simplicidad.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np

# Tipos de uso del suelo
LAND_RURAL = 0
LAND_URBAN = 1
LAND_VEG = 2
LAND_WATER = 3

LAND_NAMES = {
    LAND_RURAL: "rural",
    LAND_URBAN: "urbano",
    LAND_VEG: "vegetacion",
    LAND_WATER: "agua",
}


@dataclass
class SimulationParams:
    # Tamaño de la grilla
    nrows: int = 60
    ncols: int = 60

    # Tiempo
    steps: int = 240
    dt_hours: float = 1.0  # tamaño de paso en horas

    # Dinámica térmica
    diffusion: float = 0.25  # 0..1, mezcla con vecinos
    base_temp: float = 20.0  # °C, temperatura de referencia
    diurnal_amp: float = 6.0  # °C por hora (amplitud del ciclo diario)
    diurnal_period: float = 24.0  # horas por ciclo

    # Forzamientos locales (°C por hora)
    urban_heat: float = 0.6  # ganancia por celda urbana
    veg_cooling: float = 0.4  # enfriamiento por vegetación
    water_cooling: float = 0.8  # enfriamiento por agua

    # Ruido
    noise_std: float = 0.2  # desviación estándar del ruido por paso

    # Distribución del uso del suelo
    urban_fraction: float = 0.15
    veg_fraction: float = 0.25
    water_fraction: float = 0.05
    pattern: str = "center"  # "center" o "random"
    urban_radius_frac: float = 0.22  # si pattern="center"

    # Condiciones iniciales (°C)
    urban_init_offset: float = 1.0
    veg_init_offset: float = -0.5
    water_init_offset: float = -1.0
    rural_init_offset: float = 0.0

    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)


def _neighbor_mean(field: np.ndarray) -> np.ndarray:
    """Promedio de vecinos Moore (8 vecinos) con borde toroidal."""
    neighbors = (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, 1, axis=0), -1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), 1, axis=1)
        + np.roll(np.roll(field, -1, axis=0), -1, axis=1)
    )
    return neighbors / 8.0


def generate_land_use(params: SimulationParams) -> np.ndarray:
    """Genera un mapa de uso del suelo.

    pattern="center": crea un núcleo urbano circular y el resto se asigna
    a vegetación/agua de forma aleatoria.
    pattern="random": distribuye todo de forma aleatoria con fracciones dadas.
    """
    rng = np.random.default_rng(params.seed)
    nrows, ncols = params.nrows, params.ncols

    land = np.full((nrows, ncols), LAND_RURAL, dtype=np.int8)

    if params.pattern == "center":
        rr, cc = np.ogrid[:nrows, :ncols]
        center_r, center_c = nrows / 2.0, ncols / 2.0
        radius = params.urban_radius_frac * min(nrows, ncols)
        mask_urban = (rr - center_r) ** 2 + (cc - center_c) ** 2 <= radius ** 2
        land[mask_urban] = LAND_URBAN

        # Relleno aleatorio de vegetación/agua en zonas no urbanas
        mask_nonurban = ~mask_urban
        total_nonurban = mask_nonurban.sum()
        veg_cells = int(params.veg_fraction * nrows * ncols)
        water_cells = int(params.water_fraction * nrows * ncols)
        veg_cells = min(veg_cells, total_nonurban)
        water_cells = min(water_cells, total_nonurban - veg_cells)

        indices = np.flatnonzero(mask_nonurban)
        rng.shuffle(indices)
        veg_idx = indices[:veg_cells]
        water_idx = indices[veg_cells:veg_cells + water_cells]
        land.flat[veg_idx] = LAND_VEG
        land.flat[water_idx] = LAND_WATER
    else:
        # Asignación aleatoria completa
        probs = np.array([
            1.0 - params.urban_fraction - params.veg_fraction - params.water_fraction,
            params.urban_fraction,
            params.veg_fraction,
            params.water_fraction,
        ])
        if np.any(probs < 0):
            raise ValueError("Las fracciones de uso del suelo exceden 1.0")
        land = rng.choice([LAND_RURAL, LAND_URBAN, LAND_VEG, LAND_WATER], size=(nrows, ncols), p=probs)

    return land


def initialize_temperature(land_use: np.ndarray, params: SimulationParams) -> np.ndarray:
    """Inicializa temperatura base con offsets por tipo de suelo."""
    T = np.full_like(land_use, params.base_temp, dtype=np.float64)
    T = T + (land_use == LAND_URBAN) * params.urban_init_offset
    T = T + (land_use == LAND_VEG) * params.veg_init_offset
    T = T + (land_use == LAND_WATER) * params.water_init_offset
    T = T + (land_use == LAND_RURAL) * params.rural_init_offset
    return T


def diurnal_cycle(time_hours: float, amp: float, period: float) -> float:
    """Ciclo diario senoidal. Devuelve °C por hora."""
    return amp * np.sin(2.0 * np.pi * time_hours / period)


def local_forcing(land_use: np.ndarray, time_hours: float, params: SimulationParams) -> np.ndarray:
    """Forzamiento local neto (°C por hora)."""
    diurnal = diurnal_cycle(time_hours, params.diurnal_amp, params.diurnal_period)
    forcing = np.full_like(land_use, diurnal, dtype=np.float64)
    forcing += (land_use == LAND_URBAN) * params.urban_heat
    forcing -= (land_use == LAND_VEG) * params.veg_cooling
    forcing -= (land_use == LAND_WATER) * params.water_cooling
    return forcing


def simulate(params: SimulationParams, store_every: int = 0) -> Dict:
    """Ejecuta la simulación.

    store_every: si >0, guarda snapshots cada N pasos.
    """
    rng = np.random.default_rng(params.seed)
    land_use = generate_land_use(params)
    T = initialize_temperature(land_use, params)

    # Precalcular máscaras para métricas
    mask_urban = land_use == LAND_URBAN
    mask_rural = land_use == LAND_RURAL
    mask_veg = land_use == LAND_VEG
    mask_water = land_use == LAND_WATER

    times: List[float] = []
    mean_all: List[float] = []
    mean_urban: List[float] = []
    mean_rural: List[float] = []
    mean_veg: List[float] = []
    mean_water: List[float] = []
    uhi: List[float] = []

    snapshots: List[Tuple[float, np.ndarray]] = []

    for step in range(params.steps):
        time_hours = step * params.dt_hours

        # Métricas antes de actualizar
        times.append(time_hours)
        mean_all.append(float(T.mean()))
        mean_urban.append(float(T[mask_urban].mean()) if mask_urban.any() else np.nan)
        mean_rural.append(float(T[mask_rural].mean()) if mask_rural.any() else np.nan)
        mean_veg.append(float(T[mask_veg].mean()) if mask_veg.any() else np.nan)
        mean_water.append(float(T[mask_water].mean()) if mask_water.any() else np.nan)

        nonurban_mask = mask_rural | mask_veg | mask_water
        if mask_urban.any() and nonurban_mask.any():
            uhi.append(float(T[mask_urban].mean() - T[nonurban_mask].mean()))
        else:
            uhi.append(np.nan)

        if store_every > 0 and step % store_every == 0:
            snapshots.append((time_hours, T.copy()))

        # Dinámica
        neighbor_mean = _neighbor_mean(T)
        forcing = local_forcing(land_use, time_hours, params)
        noise = rng.normal(0.0, params.noise_std, size=T.shape)

        T = T + params.diffusion * (neighbor_mean - T) + forcing * params.dt_hours + noise

    results = {
        "params": params,
        "land_use": land_use,
        "T_last": T,
        "metrics": {
            "time_hours": np.array(times),
            "mean_all": np.array(mean_all),
            "mean_urban": np.array(mean_urban),
            "mean_rural": np.array(mean_rural),
            "mean_veg": np.array(mean_veg),
            "mean_water": np.array(mean_water),
            "uhi": np.array(uhi),
        },
        "snapshots": snapshots,
    }
    return results
