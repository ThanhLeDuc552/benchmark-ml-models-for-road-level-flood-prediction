# Terrain Feature Extraction (50m Buffer)

## Overview
This module derives all terrain features specified in `feature_table.xlsx` (`feature_dictionary` sheet) under `parent_group = 'terrain'`, `researched = 1`, and `phase_1_inclusion = 1`:
- **`elev_min`**: Minimum DEM elevation within a 50m buffer corridor around the road segment (meters).
- **`elev_mean`**: Mean DEM elevation within the 50m buffer corridor (meters).
- **`slope_mean`**: Mean topographic slope within the 50m buffer corridor (degrees).
- **`tpi`**: Mean Topographic Position Index within the 50m buffer corridor (meters).
- **`depression_depth`**: Maximum depression / sink depth within the 50m buffer corridor (meters).

## Input Datasets
- **`COPERNICUS_DANANG_FULL_EPSG2045_clean.tif`**: 30m Copernicus GLO-30 DEM projected in `EPSG:2045`.
- **`depression_depth.tif`**: Closed-low sink depth raster (filled DEM minus DEM) in `EPSG:2045`.
- **`tpi.tif`**: Multi-scale Topographic Position Index raster in `EPSG:2045`.
- **Road Network**: `hai_chau_road_graph.gpkg` (segmented road graph, <= 100m segments).

## Methodology
1. **Coordinate Projection**: Road segments are projected to `EPSG:2045` (Hanoi 1972 / GK Zone 19).
2. **50m Buffering**: Each road centerline is buffered by 50 meters Euclidean distance (`geom.buffer(50.0)`), creating a 100m corridor capturing local road embankment and drainage profile.
3. **Slope Calculation**: Slope in degrees is computed across the DEM using Horn's (1981) 3x3 finite-difference algorithm, matching GDAL / QGIS standard slope processing.
4. **Zonal Extraction**: For each 50m polygon buffer, overlapping raster cells (`all_touched=True`) are masked out and zonal statistics are computed:
   - `elev_min`: `min()` of valid DEM cells
   - `elev_mean`: `mean()` of valid DEM cells
   - `slope_mean`: `mean()` of valid slope cells (degrees)
   - `tpi`: `mean()` of valid TPI cells
   - `depression_depth`: `max()` of valid depression depth cells

## Output
Results are cached as `terrain_features_cache.csv` and can be imported directly into the dataset pipeline.

## Usage
Run standalone:
```powershell
conda run -n flood-gis python scripts/terrain_data/build_terrain_data.py
```
