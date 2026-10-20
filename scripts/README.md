# Da Nang Flood Prediction - Feature Engineering Scripts

This repository contains modular GIS and feature engineering scripts designed to derive hydraulic, terrain, rainfall, network, and ground truth passability features for the Hai Chau road network in Da Nang, projected in **EPSG:2045** (Hanoi 1972 / GK Zone 19).

## Module Directory Structure

```
scripts/
├── build_phase1_features.py        # Master orchestrator pipeline for phase-1 features
├── ground_truth/
│   ├── build_passability.py        # Derives road passability (0, 1, 2) from HEC-RAS 2D simulation
│   ├── passability_summary.csv     # Segment-level passability metrics and depth
│   └── README.md
├── hydrology_data/
│   ├── build_river_data.py         # Extracts river network from HDF, computes dist_river
│   ├── danang_rivers_epsg2045.gpkg # Extracted river centerline vector layer
│   └── README.md
├── terrain_data/
│   ├── build_terrain_data.py       # Zonal stats on 50m buffers (elev, slope, tpi, depression)
│   ├── terrain_features_cache.csv  # Cached terrain feature values
│   └── README.md
├── rainfall_data/
│   ├── build_rainfall_data.py      # HEC-DSS time-series extraction & nearest spatial join
│   ├── station_rainfall_summary.csv # Station-level rainfall metrics
│   └── README.md
└── road_data/
    ├── hai_chau_road_data.py       # Base network extraction, splitting (<=100m) & graph generation
    └── README.md
```

## Feature Specifications & Derivation

| Column | Parent Group | Role | Unit / Dtype | Derivation Method | Encoded in GraphML? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `segment_id` | `road` | `id` | String | Unique segmented edge ID | Yes |
| `osm_u` | `road` | `id` | Int | OSM start intersection node ID | Yes |
| `osm_v` | `road` | `id` | Int | OSM end intersection node ID | Yes |
| `osm_key` | `road` | `id` | Int | Parallel edge key | Yes |
| `node_degree` | `road` | `feature` | Count | Node connectivity degree | Yes |
| `length_m` | `road` | `feature` | Meters | Segment length (split at <= 100 m) | Yes |
| `road_class` | `road` | `feature` | Category | OSM highway classification | Yes |
| `is_underpass_tunnel_bridge` | `road` | `feature` | 0/1 | Structural bridge/tunnel flag | Yes |
| `dist_river` | `hydrology` | `feature` | Meters | Euclidean distance to Han River / network | Yes |
| `elev_min` | `terrain` | `feature` | Meters | Min DEM elevation within 50m buffer | Yes |
| `elev_mean` | `terrain` | `feature` | Meters | Mean DEM elevation within 50m buffer | Yes |
| `slope_mean` | `terrain` | `feature` | Degrees | Mean Horn's slope within 50m buffer | Yes |
| `tpi` | `terrain` | `feature` | Meters | Mean Topographic Position Index in 50m buffer | Yes |
| `depression_depth` | `terrain` | `feature` | Meters | Max depression depth in 50m buffer | Yes |
| `rain_total` | `rainfall` | `feature` | mm | Event total rainfall (nearest station DSS) | **No** (excluded by rule) |
| `rain_peak_intensity` | `rainfall` | `feature` | mm/h | Peak hourly rainfall (nearest station DSS) | **No** (excluded by rule) |
| `geometry_wkt` | `id` | `id` | WKT Line | LineString geometry in `EPSG:2045` | Handled via `geometry` attr |
| `passability` | `ground_truth`| `target` | `0/1/2` | Peak water depth from HEC-RAS 2D model | **Yes** |

### Passability Classification Criteria
- **`0: passable`**: $0 - 10\text{ cm}$ ($\text{depth} < 0.10\text{ m}$)
- **`1: flooded - passable`**: $10 - 30\text{ cm}$ ($0.10\text{ m} \le \text{depth} \le 0.30\text{ m}$)
- **`2: non-passable`**: $> 30\text{ cm}$ ($\text{depth} > 0.30\text{ m}$)
