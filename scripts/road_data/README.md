# Road Data Extraction & Processing

This folder contains scripts for extracting and processing road network, transportation, and infrastructure spatial data for the **Da Nang Flood Prediction System**.

---

## 1. Script Overview: `extract_danang_roads.py`

Extracts OpenStreetMap (OSM) vector layers from the regional Vietnam GeoPackage (`vietnam.gpkg`) filtered/clipped by the administrative boundary in Da Nang GeoJSON (`Đà Nẵng (tỉnh thành) - 34_simplified.geojson`).

The extracted data is exported as an optimized GeoPackage (`danang.gpkg`) stored alongside the source dataset in `Data Collections/extras/vietnam-261008-free.gpkg/`.

---

## 2. Default File Paths

| Role | Path |
| :--- | :--- |
| **Source GeoPackage** | `Data Collections/extras/vietnam-261008-free.gpkg/vietnam.gpkg` |
| **Boundary GeoJSON** | `Data Collections/Đà Nẵng (tỉnh thành) - 34_simplified.geojson` |
| **All-Layer GeoPackage** | `Data Collections/extras/vietnam-261008-free.gpkg/danang.gpkg` |
| **Major Arterial & Highway GPKG** | `Data Collections/extras/vietnam-261008-free.gpkg/danang_roads.gpkg` |

---

## 3. Usage & CLI Options

### Basic Usage (Extract Default 8 Major Road Corridors)
```powershell
python "Scripts/road_data/extract_danang_roads.py"
```
*Extracts only the 8 major arterial, highway, and link road types (4,608 segments, 2,668.6 km): `motorway`, `trunk`, `primary`, `secondary`, and their respective links (`*_link`).*

### Extract All Road Types (Unfiltered OSM Roads)
```powershell
python "Scripts/road_data/extract_danang_roads.py" --all-road-types
```
*Extracts all `gis_osm_roads_free` (80,728 segments) without filtering.*

### Extract Custom Road Types
```powershell
python "Scripts/road_data/extract_danang_roads.py" --road-types "motorway,trunk,primary"
```

### Extract All Road & Transport Layers
```powershell
python "Scripts/road_data/extract_danang_roads.py" --all-road-layers
```
*Extracts roads, railways, traffic signals/stops, traffic areas, transit points, and transit polygons (90,873 features).*

### Extract All 20 OSM Layers
```powershell
python "Scripts/road_data/extract_danang_roads.py" --all-layers
```

### Exact Boundary Clipping (clean cut at boundary)
```powershell
python "Scripts/road_data/extract_danang_roads.py" --clip-mode exact
```

### Extract Specific Layers or Road Types
```powershell
python "Scripts/road_data/extract_danang_roads.py" --layers "gis_osm_roads_free,gis_osm_waterways_free"
python "Scripts/road_data/extract_danang_roads.py" --road-types "primary,secondary,trunk,motorway"
```

### List Available Source Layers
```powershell
python "Scripts/road_data/extract_danang_roads.py" --list-layers
```

---

## 4. CLI Arguments Reference

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--gpkg` | `vietnam.gpkg` | Path to source Vietnam GeoPackage. |
| `--boundary` | `Đà Nẵng...geojson` | Path to administrative boundary polygon. |
| `--output` | `danang_roads.gpkg` | Path for the output GeoPackage. |
| `--layers` | `gis_osm_roads_free` | Comma-separated list of layer names to extract. |
| `--all-road-layers` | `False` | Extract roads, railways, traffic, and transit layers. |
| `--all-layers` | `False` | Extract all 20 OSM layers for Da Nang. |
| `--clip-mode` | `intersect` | `intersect` (preserve full line features touching boundary), `exact` (clip geometries cleanly at polygon boundary), or `bbox` (bounding box query). |
| `--no-overwrite` | `False` | Do not overwrite existing output file. |
| `--list-layers` | `False` | Print available layers in source GeoPackage and exit. |

---

## 5. Exploratory Data Analysis: `road_analysis.ipynb`

A Jupyter notebook providing comprehensive exploratory data analysis of Da Nang's road infrastructure:
- **Layer & Metadata Profiling**: All 6 road & transportation layers inspected with schemas and feature counts.
- **Road Type (`fclass`) Analysis**: Breakdown of 26+ OSM road classifications by segment frequency, mileage (km), and distributions.
- **Metric Projections**: Length calculations reprojected to UTM Zone 49N (`EPSG:32649`).
- **Functional Hierarchy**: Categorization into 7 functional tiers (Freeways, Arterials, Collectors, Residential, Tracks, Service, Pedestrian).
- **Flood Risk & Infrastructure**: Cross-analysis of bridges (2,735 features), tunnels, one-way street constraints, and national highway corridors (`ref`).
- **Spatial Maps**: Multi-tier hierarchy maps, bridge crossing overlays, and interactive Leaflet map (`danang_roads_interactive.html`).

---

## 6. Hai Chau Road Graph & Feature Table Derivation: `hai_chau_road_data.py`

Derives the standardized segment-level road graph and populates the `feature_table` sheet in `feature_table.xlsx` for features satisfying `"parent_group" == "road" and "researched" && "phase_1_inclusion" == 1`.

### Data Sources
- **Administrative Boundary**: `hai_chau.geojson`
- **Da Nang GIS Layers**: `danang.gpkg` (layer `gis_osm_roads_free`)
- **Base Graph / Cache**: `hai_chau_network.graphml` / OSMnx cache (`cache/`)

### Processing Rules & Derivations
1. **Administrative Filtering**: Truncates the drivable road network precisely to the Hai Chau polygon.
2. **Infrastructure Tagging**: Cross-references OSM IDs and geometries against `danang.gpkg` to identify bridges, tunnels, and underpasses (`is_underpass_tunnel_bridge`).
3. **Intersection & Metric Splitting**: Splits edges at network intersections and ensures every segment has length $\le 100\text{ m}$ (calculated in `EPSG:2045`).
4. **Graph Encoding**: Encodes `segment_id`, `osm_u`, `osm_v`, `osm_key`, `node_degree`, `length_m`, `road_class`, `is_underpass_tunnel_bridge`, and geometry onto every segment.
5. **Output Graph Formats**:
   - `hai_chau_road_graph.graphml`: GraphML topological network graph with all node and edge attributes.
   - `hai_chau_road_graph.gpkg`: OGC GeoPackage vector layer (`hai_chau_road_segments`) for GIS visualization and spatial joins.
6. **Feature Table Occupation**: Fills the derived road features into `feature_table.xlsx` (`feature_table` sheet) while strictly preserving all other columns untouched.

### Execution
```powershell
python "scripts/road_data/hai_chau_road_data.py"
```

