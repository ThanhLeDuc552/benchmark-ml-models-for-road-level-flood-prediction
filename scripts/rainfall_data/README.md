# Rainfall Feature Extraction (`rain_total`, `rain_peak_intensity`)

## Overview
This module derives the rainfall features specified in `feature_table.xlsx` (`feature_dictionary` sheet) under `parent_group = 'rainfall'`, `researched = 1`, and `phase_1_inclusion = 1`:
- **`rain_total`**: Event total precipitation assigned to each road segment (mm).
- **`rain_peak_intensity`**: Peak 1-hour rainfall intensity assigned to each road segment (mm/h).

## Input Data
- **Station Directory (`new DSSs/`)**: Contains 428 rain stations (`station_0000` to `station_0427`).
- **Station Metadata (`new DSSs/stations.csv`)**: Contains station coordinates in both geographic (`latitude`, `longitude`) and projected metric coordinates (`project_x_epsg_2045`, `project_y_epsg_2045`).
- **Precipitation Records (`new DSSs/<station>/precipitation.dss`)**: HEC-DSS regular time series containing hourly precipitation observations (`1Hour/OBS`).

## Methodology
1. **HEC-DSS Time-Series Parsing**: Using `pydsstools`, the hourly precipitation time-series for the October 2022 event record (`01Oct2022/1Hour/OBS/`) is extracted across all stations.
2. **Event Metrics Computation**:
   - `rain_total` = Sum of all positive hourly observations across the event month record (mm).
   - `rain_peak_intensity` = Maximum hourly observation (mm/h), matching the October 14, 2022 storm peak.
3. **Spatial Join (Nearest Station)**:
   - Built a 2D spatial KD-Tree (`scipy.spatial.cKDTree`) on the projected coordinates (`EPSG:2045`) of the stations.
   - For each road segment in Hai Chau (represented by its centroid in `EPSG:2045`), the nearest station is identified and its rainfall metrics are assigned.
   - Nearest stations for Hai Chau road segments average ~165 meters in distance (min 6.2 m, max 558 m), with 65 distinct stations mapped to the network.

## Graph Encoding Note
As specified by dataset specifications, features with `parent_group == "rainfall"` (`rain_total`, `rain_peak_intensity`) are **NOT** encoded into `hai_chau_road_graph.graphml`, but are **fully occupied** in the `feature_table` sheet of `feature_table.xlsx`.

## Usage
Run standalone:
```powershell
conda run -n flood-gis python scripts/rainfall_data/build_rainfall_data.py
```
