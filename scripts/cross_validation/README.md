# Spatial Cross-Validation: Spatial Grid Blocks (`block_id`)

## Overview
This module partitions the study area into $500\text{ m} \times 500\text{ m}$ square spatial blocks (`block_id`) based on road segment centroids in **EPSG:2045** (Hanoi 1972 / GK Zone 19).

As specified in `feature_table.xlsx` under `parent_group = 'road'`, `role = 'id'`, `child_group = 'cross-validation'`:
- Blocks prevent spatial autocorrelation leakage between adjacent training and testing road segments.
- Blocks are subsequently assigned to 5 spatial folds for **Spatial Block Cross-Validation** to quantify the Optimism Gap ($\Delta_m$).

## Methodology
1. **Centroid Extraction**: Segment line strings in `geometry_wkt` are parsed to extract geometric centroids $(c_x, c_y)$ in metres.
2. **2D Grid Binning**: Coordinates are binned into $500\text{ m}$ grid cells:
   $$b_x = \lfloor (c_x - c_{x,\min}) / 500 \rfloor, \quad b_y = \lfloor (c_y - c_{y,\min}) / 500 \rfloor$$
   $$\text{block\_id} = f"{b_x}\_{b_y}"$$

## Target Files Updated
- **`feature_table.xlsx`**: Occupies `block_id` column in sheet `feature_table`.
- **`hai_chau_road_graph.graphml`**: Encodes `block_id` attribute on all network edges.
- **`hai_chau_road_graph.gpkg`**: Adds `block_id` column to the GeoPackage layer.

## Usage
Run standalone:
```powershell
conda run -n flood-gis python scripts/cross_validation/build_spatial_blocks.py
```
