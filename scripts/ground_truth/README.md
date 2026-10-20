# Ground Truth Feature Extraction: Road Passability (`passability`)

## Overview
This module derives the ground truth target feature `passability` (specified in `feature_table.xlsx` under `parent_group = 'ground_truth'`, `role = 'target'`, `phase_1_inclusion = 1`) based on the hydraulic flood simulation run from HEC-RAS (`2022-10-14_to_2022-10-15`) and the road segment geometry (`geometry_wkt` in `EPSG:2045`).

## Classification Categories

As specified by dataset requirements:
- **`0: passable`**: Water level $0 - 10\text{ cm}$ ($\text{depth} < 0.10\text{ m}$)
- **`1: flooded - passable`**: Water level $10 - 30\text{ cm}$ ($0.10\text{ m} \le \text{depth} \le 0.30\text{ m}$)
- **`2: non-passable`**: Water level $> 30\text{ cm}$ ($\text{depth} > 0.30\text{ m}$)

## Methodology
1. **HEC-RAS 2D Mesh Extraction**: Peak water inundation depth across all 37 hourly simulation timesteps is extracted from `2022-10-14_to_2022-10-15/danangfloodmodel.p05.hdf` across Perimeter 1 and Perimeter 2:
   $$\text{depth}_i = \max_t \Big(\max(0, \text{WSE}_{i, t} - \text{Elev}_i)\Big)$$
2. **Dense Road Sampling**: For each segment in `geometry_wkt`, points are sampled every 10 meters along the centerline to capture elevation dips and bottleneck inundation points.
3. **Nearest-Neighbor Spatial Query**: A 2D spatial KD-Tree (`scipy.spatial.cKDTree`) maps each sample point to its corresponding 2D hydraulic mesh cell.
4. **Classification**: The critical bottleneck water depth determines the segment's passability class (`0`, `1`, or `2`).

## Target Files Updated
- **`feature_table.xlsx`**: Occupies `passability` column in sheet `feature_table`.
- **`hai_chau_road_graph.graphml`**: Encodes `passability` attribute on all network edges.
- **`hai_chau_road_graph.gpkg`**: Adds `passability` column to the GeoPackage layer.

## Usage
Run standalone:
```powershell
conda run -n flood-gis python scripts/ground_truth/build_passability.py
```
