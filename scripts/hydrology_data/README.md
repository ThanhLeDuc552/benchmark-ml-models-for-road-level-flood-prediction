# Hydrology Data Feature Extraction (`dist_river`)

## Overview
This module derives the `dist_river` feature (defined in `feature_dictionary` under `parent_group = 'hydrology'`, `researched = 1`, `phase_1_inclusion = 1`) for the Hai Chau road network.

## Source Data
- **HEC-RAS Geometry File**: `danangfloodmodel.g01.hdf`
- **Native CRS**: Hanoi 1972 / Gauss-Kruger Zone 19 (`EPSG:2045`), matching the project projection standard.
- **Rivers Extracted**:
  - Han River (`Han`, Main Reach)
  - Cam Le River (`Cam Le`, Main Reach)
  - Yen River (`Yen`, Main Reach)
  - Tuy Loan River (`Tuy Loan`, Main Reach)
  - Ai Nghia River (`Ai Nghia`, Main Reach)
  - Lac Thanh River (`Lac Thanh`, Main Reach)
  - Vu Gia River (`Vu Gia`, Main Reach)

## Methodology
1. **River Geometry Extraction**: River polyline coordinates and attributes are extracted from HDF group `Geometry/River Centerlines`.
2. **Intermediate Product**: Stored as `danang_rivers_epsg2045.gpkg` for GIS visual verification and cartographic reference.
3. **Distance Calculation**: For each segmented road segment in Hai Chau (reprojected to `EPSG:2045`), the minimum Euclidean distance (meters) to the river network is computed. Because Hai Chau is located along the Han River, the nearest river line for all segments is the Han River line.

## Usage
Run standalone:
```powershell
conda run -n flood-gis python scripts/hydrology_data/build_river_data.py
```
Or import `extract_rivers_from_hdf` and `compute_dist_river` in python workflows.
