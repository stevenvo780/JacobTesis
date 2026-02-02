"""CLI para ejecutar la simulación de isla de calor urbano con autómatas celulares."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import numpy as np

from src.ca_model import SimulationParams, simulate
from src.viz import plot_land_use, plot_temperature, plot_uhi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulación CA de isla de calor urbano")

    # Tamaño y tiempo
    p.add_argument("--nrows", type=int, default=60)
    p.add_argument("--ncols", type=int, default=60)
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--dt-hours", type=float, default=1.0)

    # Dinámica
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

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--store-every", type=int, default=0)
    p.add_argument("--output", type=str, default="output/run")

    return p.parse_args()


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

    results = simulate(params, store_every=args.store_every)

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Guardar configuración
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(params), f, indent=2, ensure_ascii=False)

    # Guardar métricas
    metrics = results["metrics"]
    data = np.column_stack([
        metrics["time_hours"],
        metrics["mean_all"],
        metrics["mean_urban"],
        metrics["mean_rural"],
        metrics["mean_veg"],
        metrics["mean_water"],
        metrics["uhi"],
    ])
    header = "time_hours,mean_all,mean_urban,mean_rural,mean_veg,mean_water,uhi"
    np.savetxt(os.path.join(out_dir, "metrics.csv"), data, delimiter=",", header=header, comments="")

    # Guardar estado final para análisis avanzado
    np.save(os.path.join(out_dir, "temperature_final.npy"), results["T_last"])
    np.save(os.path.join(out_dir, "land_use.npy"), results["land_use"])

    # Gráficas
    plot_land_use(results["land_use"], os.path.join(out_dir, "land_use.png"))
    plot_temperature(results["T_last"], os.path.join(out_dir, "temperature_final.png"))
    plot_uhi(metrics["time_hours"], metrics["uhi"], os.path.join(out_dir, "uhi_timeseries.png"))

    print(f"OK: resultados en {out_dir}")


if __name__ == "__main__":
    main()
