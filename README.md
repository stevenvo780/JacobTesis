# MVP: Autómata Celular para Isla de Calor Urbano (UHI)

Este MVP simula una **isla de calor urbano** con un **autómata celular** 2D. La UHI se interpreta como un **rastro local** del hiperobjeto del cambio climático: un patrón térmico persistente que emerge de decisiones urbanas y del entorno físico.

El objetivo es **entender, analizar y contrastar escenarios** de urbanización, vegetación y agua, con una simulación simple pero útil y extensible.

## Qué hace el modelo (en términos simples)
Cada celda tiene un **tipo de suelo** y una **temperatura**. En cada paso:
1. La temperatura se **difunde** hacia los vecinos (se suaviza espacialmente).
2. Se aplica un **forzamiento local**:
   - Urbano calienta.
   - Vegetación y agua enfrían.
3. Se suma un **ciclo diurno** (senoidal).
4. Se añade **ruido** para variabilidad.

## Estructura del proyecto
- `src/ca_model.py`: modelo y simulación.
- `src/analysis.py`: métricas básicas.
- `src/viz.py`: gráficos.
- `run_simulation.py`: CLI para ejecutar y guardar resultados.

## Requisitos
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar
```bash
python run_simulation.py --steps 240 --nrows 60 --ncols 60 --output output/demo
```

Salidas en `output/demo`:
- `config.json`: parámetros usados.
- `metrics.csv`: series temporales de métricas.
- `temperature_final.npy`: temperatura final (NumPy).
- `land_use.npy`: uso de suelo (NumPy).
- `land_use.png`: mapa de suelo.
- `temperature_final.png`: temperatura final.
- `uhi_timeseries.png`: serie temporal de UHI.

## Resultados de la primera iteración (23 de enero de 2026)
Ejecución realizada en `output/iter1` con:
- `nrows=40`, `ncols=40`, `steps=48`, `dt_hours=1.0`
- `pattern=center`, `urban_radius_frac=0.22`
- `urban_heat=0.6`, `veg_cooling=0.4`, `water_cooling=0.8`
- `diffusion=0.25`, `diurnal_amp=6.0`, `noise_std=0.2`, `seed=42`

**Resumen UHI (°C):**
- media: 16.12
- mínimo: 1.21
- máximo: 29.63

**Último paso (t=47 h):**
- mean_all: 19.18 °C
- mean_urban: 44.35 °C
- mean_rural: 15.61 °C
- mean_veg: 13.53 °C
- mean_water: 10.87 °C
- uhi: 29.63 °C

## Parámetros clave (qué significan)
**Espacio/tiempo**
- `nrows`, `ncols`: tamaño de la grilla.
- `steps`: número de pasos.
- `dt-hours`: duración de cada paso (horas).

**Dinámica térmica**
- `diffusion`: mezcla con vecinos (0..1). Valores altos suavizan mucho.
- `base-temp`: temperatura base (°C).
- `diurnal-amp`: amplitud del ciclo diario (°C por hora).
- `diurnal-period`: periodo del ciclo (horas, típicamente 24).

**Forzamientos locales (°C por hora)**
- `urban-heat`: calor extra en celdas urbanas.
- `veg-cooling`: enfriamiento por vegetación (se resta).
- `water-cooling`: enfriamiento por agua (se resta).

**Ruido**
- `noise-std`: variabilidad aleatoria por paso.

**Uso del suelo**
- `urban-fraction`, `veg-fraction`, `water-fraction`: fracciones si `pattern=random`.
- `pattern`: `center` crea un núcleo urbano; `random` distribuye aleatorio.
- `urban-radius-frac`: radio urbano relativo si `pattern=center`.

**Iniciales**
- `urban-init-offset`, `veg-init-offset`, `water-init-offset`, `rural-init-offset`: offsets iniciales por tipo.

## Métricas (para analizar y contrastar)
En `metrics.csv`:
- `mean_all`: promedio total.
- `mean_urban`: promedio en celdas urbanas.
- `mean_rural`, `mean_veg`, `mean_water`: promedios por tipo.
- `uhi`: **UHI = mean_urban - mean_no_urban** (rural+veg+agua).

## Contrastar escenarios (experimentos rápidos)
1. **Urbanización agresiva**
```bash
python run_simulation.py --urban-fraction 0.35 --pattern random --output output/urbano_intenso
```
2. **Verde urbano fuerte**
```bash
python run_simulation.py --veg-fraction 0.45 --veg-cooling 0.7 --output output/verde
```
3. **Comparar con/ sin agua**
```bash
python run_simulation.py --water-fraction 0.0 --output output/sin_agua
python run_simulation.py --water-fraction 0.12 --output output/con_agua
```

Luego compara `uhi_timeseries.png` y los promedios de `metrics.csv`.

## Notas técnicas útiles
- **Condiciones de borde**: el borde es toroidal (la grilla “se enrolla”).
- **Escala**: este modelo es conceptual, no meteorológico real.
- **Interpretación**: el patrón UHI es un rastro local del hiperobjeto climático; sirve para explorar mecanismos y sensibilidad, no para predicción.

## Próximos pasos sugeridos
- Calibrar con datos reales (si existen).
- Añadir albedo, humedad y viento explícitos.
- Incorporar escenarios estacionales.
