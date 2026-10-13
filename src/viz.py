"""Visualización básica de resultados."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .ca_model import LAND_RURAL, LAND_URBAN, LAND_VEG, LAND_WATER, LAND_NAMES


def plot_land_use(land_use: np.ndarray, path: str) -> None:
    colors = ["#c2b280", "#444444", "#3c8d0d", "#2a6fbb"]
    cmap = ListedColormap(colors)
    plt.figure(figsize=(5, 5))
    plt.imshow(
        land_use, cmap=cmap, interpolation="nearest",
        vmin=LAND_RURAL - 0.5, vmax=LAND_WATER + 0.5,
    )
    plt.title("Uso del suelo")
    cbar = plt.colorbar(ticks=[LAND_RURAL, LAND_URBAN, LAND_VEG, LAND_WATER])
    cbar.ax.set_yticklabels([
        LAND_NAMES[LAND_RURAL],
        LAND_NAMES[LAND_URBAN],
        LAND_NAMES[LAND_VEG],
        LAND_NAMES[LAND_WATER],
    ])
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_temperature(T: np.ndarray, path: str, title: str = "Temperatura final") -> None:
    plt.figure(figsize=(5, 5))
    im = plt.imshow(T, cmap="inferno", interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, label="°C")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_uhi(time_hours: np.ndarray, uhi: np.ndarray, path: str) -> None:
    plt.figure(figsize=(6, 3.5))
    plt.plot(time_hours, uhi, color="#b00020", linewidth=1.6)
    plt.title("Intensidad UHI (urbano - no urbano)")
    plt.xlabel("Tiempo (h)")
    plt.ylabel("°C")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
