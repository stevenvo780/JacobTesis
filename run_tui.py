"""Visualizacion en terminal de la simulacion CA-UHI.

Muestra un mapa ASCII de temperatura que evoluciona en el tiempo, junto con
metricas basicas (promedios y UHI). No requiere matplotlib.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Tuple

import numpy as np

from src.ca_model import (
    SimulationParams,
    generate_land_use,
    initialize_temperature,
    local_forcing,
    _neighbor_mean,
    LAND_RURAL,
    LAND_URBAN,
    LAND_VEG,
    LAND_WATER,
)


ASCII_GRADIENT = " .:-=+*#%@"  # 10 niveles, ASCII puro


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TUI para simulacion CA-UHI")

    # Tamano y tiempo
    p.add_argument("--nrows", type=int, default=30)
    p.add_argument("--ncols", type=int, default=60)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--dt-hours", type=float, default=1.0)

    # Dinamica
    p.add_argument("--diffusion", type=float, default=0.25)
    p.add_argument("--base-temp", type=float, default=20.0)
    p.add_argument("--diurnal-amp", type=float, default=6.0)
    p.add_argument("--diurnal-period", type=float, default=24.0)
    p.add_argument("--urban-heat", type=float, default=0.6)
    p.add_argument("--veg-cooling", type=float, default=0.4)
    p.add_argument("--water-cooling", type=float, default=0.8)
    p.add_argument("--noise-std", type=float, default=0.2)

    # Uso del suelo
    p.add_argument("--urban-fraction", type=float, default=0.15)
    p.add_argument("--veg-fraction", type=float, default=0.25)
    p.add_argument("--water-fraction", type=float, default=0.05)
    p.add_argument("--pattern", type=str, default="center", choices=["center", "random"])
    p.add_argument("--urban-radius-frac", type=float, default=0.22)

    # Iniciales
    p.add_argument("--urban-init-offset", type=float, default=1.0)
    p.add_argument("--veg-init-offset", type=float, default=-0.5)
    p.add_argument("--water-init-offset", type=float, default=-1.0)
    p.add_argument("--rural-init-offset", type=float, default=0.0)

    # UI
    p.add_argument("--sleep", type=float, default=0.05, help="segundos entre frames")
    p.add_argument("--no-clear", action="store_true", help="no limpiar pantalla")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def step_temperature(
    T: np.ndarray,
    land_use: np.ndarray,
    step_idx: int,
    params: SimulationParams,
    rng: np.random.Generator,
) -> np.ndarray:
    time_hours = step_idx * params.dt_hours
    neighbor_mean = _neighbor_mean(T)
    forcing = local_forcing(land_use, time_hours, params)
    noise = rng.normal(0.0, params.noise_std, size=T.shape)
    T = T + params.diffusion * (neighbor_mean - T) + forcing * params.dt_hours + noise
    return T


def compute_metrics(T: np.ndarray, land_use: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    mask_urban = land_use == LAND_URBAN
    mask_rural = land_use == LAND_RURAL
    mask_veg = land_use == LAND_VEG
    mask_water = land_use == LAND_WATER

    mean_all = float(T.mean())
    mean_urban = float(T[mask_urban].mean()) if mask_urban.any() else float("nan")
    mean_rural = float(T[mask_rural].mean()) if mask_rural.any() else float("nan")
    mean_veg = float(T[mask_veg].mean()) if mask_veg.any() else float("nan")
    mean_water = float(T[mask_water].mean()) if mask_water.any() else float("nan")

    nonurban = mask_rural | mask_veg | mask_water
    if mask_urban.any() and nonurban.any():
        uhi = float(T[mask_urban].mean() - T[nonurban].mean())
    else:
        uhi = float("nan")

    return mean_all, mean_urban, mean_rural, mean_veg, mean_water, uhi


def render_ascii(T: np.ndarray) -> str:
    tmin = float(np.min(T))
    tmax = float(np.max(T))
    denom = (tmax - tmin) if (tmax - tmin) != 0.0 else 1.0
    scaled = (T - tmin) / denom
    idx = (scaled * (len(ASCII_GRADIENT) - 1)).astype(np.int32)

    lines = ["".join(ASCII_GRADIENT[i] for i in row) for row in idx]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    params = SimulationParams(
        nrows=args.nrows,
        ncols=args.ncols,
        steps=args.steps,
        dt_hours=args.dt_hours,
        diffusion=args.diffusion,
        base_temp=args.base_temp,
        diurnal_amp=args.diurnal_amp,
        diurnal_period=args.diurnal_period,
        urban_heat=args.urban_heat,
        veg_cooling=args.veg_cooling,
        water_cooling=args.water_cooling,
        noise_std=args.noise_std,
        urban_fraction=args.urban_fraction,
        veg_fraction=args.veg_fraction,
        water_fraction=args.water_fraction,
        pattern=args.pattern,
        urban_radius_frac=args.urban_radius_frac,
        urban_init_offset=args.urban_init_offset,
        veg_init_offset=args.veg_init_offset,
        water_init_offset=args.water_init_offset,
        rural_init_offset=args.rural_init_offset,
        seed=args.seed,
    )

    rng = np.random.default_rng(params.seed)
    land_use = generate_land_use(params)
    T = initialize_temperature(land_use, params)

    for step_idx in range(params.steps):
        time_hours = step_idx * params.dt_hours
        mean_all, mean_urban, mean_rural, mean_veg, mean_water, uhi = compute_metrics(T, land_use)

        if not args.no_clear:
            sys.stdout.write("\x1b[2J\x1b[H")

        header = (
            f"Paso {step_idx}/{params.steps - 1} | t={time_hours:.1f} h | "
            f"Tmin={T.min():.2f} C Tmax={T.max():.2f} C\n"
            f"mean_all={mean_all:.2f} mean_urban={mean_urban:.2f} mean_rural={mean_rural:.2f} "
            f"mean_veg={mean_veg:.2f} mean_water={mean_water:.2f} UHI={uhi:.2f}\n"
        )
        sys.stdout.write(header)
        sys.stdout.write(render_ascii(T))
        sys.stdout.write("\n")
        sys.stdout.flush()

        T = step_temperature(T, land_use, step_idx, params, rng)
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
