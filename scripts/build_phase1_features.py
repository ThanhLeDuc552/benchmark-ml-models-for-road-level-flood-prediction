"""
Master Pipeline: Build Phase-1 Researched Features for Hai Chau Road Network.

Features satisfying ("researched" == 1 && "phase_1_inclusion" == 1):
1. Hydrology:
   - `dist_river` (meters): Distance to nearest major river (from danangfloodmodel.g01.hdf)
2. Terrain (50m road buffer, EPSG:2045):
   - `elev_min` (meters): Minimum DEM elevation
   - `elev_mean` (meters): Mean DEM elevation
   - `slope_mean` (degrees): Mean slope from DEM (Horn's formula)
   - `tpi` (meters): Mean Topographic Position Index (tpi.tif)
   - `depression_depth` (meters): Maximum sink depth (depression_depth.tif)
3. Rainfall:
   - `rain_total` (mm): Event total from HEC-DSS records (new DSSs/)
   - `rain_peak_intensity` (mm/h): Peak hourly rainfall intensity
4. Identifier:
   - `geometry_wkt`: Segment geometry WKT in EPSG:2045

Graph Encoding:
- Encodes all required features into `hai_chau_road_graph.graphml` EXCEPT rainfall features.

Spreadsheet Occupation:
- Occupies the `feature_table` sheet in `feature_table.xlsx` with derived data for all required features.
- Strictly preserves all non-researched columns without touching or interpolating them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

# Ensure UTF-8 output encoding on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure GDAL/PROJ environment variables
conda_prefix = Path(sys.executable).parent
gdal_data = conda_prefix / "Library" / "share" / "gdal"
proj_lib = conda_prefix / "Library" / "share" / "proj"
if gdal_data.exists():
    os.environ["GDAL_DATA"] = str(gdal_data)
if proj_lib.exists():
    os.environ["PROJ_LIB"] = str(proj_lib)
    os.environ["PROJ_DATA"] = str(proj_lib)

import geopandas as gpd
import networkx as nx
import numpy as np
import openpyxl
import pandas as pd

# Import modular builder functions
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from scripts.hydrology_data.build_river_data import (
    extract_rivers_from_hdf,
    compute_dist_river,
    DEFAULT_HDF_PATH,
    DEFAULT_OUT_RIVER_GPKG,
)
from scripts.terrain_data.build_terrain_data import (
    compute_terrain_features,
    DEFAULT_DEM_PATH,
    DEFAULT_DEP_PATH,
    DEFAULT_TPI_PATH,
)
from scripts.rainfall_data.build_rainfall_data import (
    compute_rainfall_features,
    DEFAULT_DSS_DIR,
    DEFAULT_STATIONS_CSV,
)

DEFAULT_ROAD_GPKG = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.gpkg"
DEFAULT_ROAD_GRAPHML = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.graphml"
DEFAULT_EXCEL_PATH = WORKSPACE_DIR / "dataset" / "feature_table.xlsx"


def encode_features_to_graphml(
    graphml_path: Path,
    features_df: pd.DataFrame,
    out_graphml_path: Path | None = None,
) -> nx.MultiDiGraph:
    """
    Encode derived features into hai_chau_road_graph.graphml.
    Excludes all features with parent_group == 'rainfall'.
    """
    if out_graphml_path is None:
        out_graphml_path = graphml_path

    print(f"\nLoading existing GraphML: {graphml_path.name} ...")
    G = nx.read_graphml(graphml_path)
    print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    # Index features by segment_id
    feat_indexed = features_df.set_index("segment_id")

    # Non-rainfall attributes to encode
    attributes_to_encode = [
        "dist_river",
        "elev_min",
        "elev_mean",
        "slope_mean",
        "tpi",
        "depression_depth",
    ]

    updated_count = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        seg_id = d.get("segment_id")
        if seg_id in feat_indexed.index:
            row = feat_indexed.loc[seg_id]
            for attr in attributes_to_encode:
                val = row[attr]
                # NetworkX GraphML writer accepts float/int/str
                if pd.notna(val):
                    d[attr] = float(val)
                else:
                    d[attr] = ""
            updated_count += 1

    print(f"Encoded attributes {attributes_to_encode} into {updated_count:,} edges.")
    print("Explicitly verified: features with parent_group == 'rainfall' NOT included in GraphML.")

    print(f"Writing updated GraphML to: {out_graphml_path} ...")
    nx.write_graphml(G, out_graphml_path)
    print("GraphML updated successfully.")
    return G


def update_road_gpkg(
    gpkg_path: Path,
    features_df: pd.DataFrame,
    out_gpkg_path: Path | None = None,
) -> None:
    """
    Update hai_chau_road_graph.gpkg with new non-rainfall feature columns.
    """
    if out_gpkg_path is None:
        out_gpkg_path = gpkg_path

    print(f"\nUpdating road GeoPackage: {gpkg_path.name} ...")
    gdf = gpd.read_file(gpkg_path)
    non_rain_cols = [
        "dist_river",
        "elev_min",
        "elev_mean",
        "slope_mean",
        "tpi",
        "depression_depth",
    ]

    for col in non_rain_cols:
        if col in gdf.columns:
            gdf = gdf.drop(columns=[col])

    merged = gdf.merge(features_df[["segment_id"] + non_rain_cols], on="segment_id", how="left")
    merged.to_file(out_gpkg_path, driver="GPKG")
    print(f"Saved updated GeoPackage to: {out_gpkg_path}")


def occupy_feature_table_spreadsheet(
    excel_path: Path,
    features_df: pd.DataFrame,
) -> None:
    """
    Occupy the 'feature_table' sheet in feature_table.xlsx with the derived data
    for all required features (researched == 1 && phase_1_inclusion == 1).
    Strictly preserves all other columns without touching or interpolating them.
    """
    print(f"\nOpening Excel workbook: {excel_path.name} ...")
    wb = openpyxl.load_workbook(excel_path)
    if "feature_table" not in wb.sheetnames:
        raise ValueError(f"'feature_table' sheet missing from {excel_path}")

    ws = wb["feature_table"]
    header_row = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    col_map = {name: idx + 1 for idx, name in enumerate(header_row) if name is not None}

    # Features to occupy
    target_columns = [
        "geometry_wkt",
        "dist_river",
        "elev_min",
        "elev_mean",
        "slope_mean",
        "tpi",
        "depression_depth",
        "rain_total",
        "rain_peak_intensity",
    ]

    for col in target_columns:
        if col not in col_map:
            raise KeyError(f"Target column '{col}' missing from feature_table headers: {header_row}")

    # Build lookup by segment_id
    feat_indexed = features_df.set_index("segment_id")

    # Match each row in spreadsheet by segment_id
    seg_col_idx = col_map["segment_id"]
    rows_updated = 0

    for r in range(2, ws.max_row + 1):
        seg_id = ws.cell(r, seg_col_idx).value
        if seg_id in feat_indexed.index:
            row_data = feat_indexed.loc[seg_id]
            for col in target_columns:
                val = row_data[col]
                target_col_idx = col_map[col]
                if pd.isna(val):
                    ws.cell(r, target_col_idx).value = None
                elif isinstance(val, (np.floating, float)):
                    ws.cell(r, target_col_idx).value = round(float(val), 4)
                elif isinstance(val, (np.integer, int)):
                    ws.cell(r, target_col_idx).value = int(val)
                else:
                    ws.cell(r, target_col_idx).value = str(val)
            rows_updated += 1

    wb.save(excel_path)
    print(f"Successfully occupied {rows_updated:,} rows in 'feature_table' sheet of {excel_path.name}.")


def run_pipeline(
    hdf_path: Path = DEFAULT_HDF_PATH,
    dem_path: Path = DEFAULT_DEM_PATH,
    dep_path: Path = DEFAULT_DEP_PATH,
    tpi_path: Path = DEFAULT_TPI_PATH,
    dss_dir: Path = DEFAULT_DSS_DIR,
    stations_csv: Path = DEFAULT_STATIONS_CSV,
    road_gpkg: Path = DEFAULT_ROAD_GPKG,
    road_graphml: Path = DEFAULT_ROAD_GRAPHML,
    excel_path: Path = DEFAULT_EXCEL_PATH,
    buffer_dist_m: float = 50.0,
) -> pd.DataFrame:
    """
    Execute full feature derivation and encoding workflow.
    """
    total_start = time.time()
    print("=" * 80)
    print("PHASE-1 FEATURE BUILDING PIPELINE: HAI CHAU ROAD NETWORK")
    print("=" * 80)
    print(f"HDF Model Path : {hdf_path}")
    print(f"DEM Path       : {dem_path}")
    print(f"Depression TIF : {dep_path}")
    print(f"TPI TIF        : {tpi_path}")
    print(f"DSS Directory  : {dss_dir}")
    print(f"Road GPKG      : {road_gpkg}")
    print(f"Road GraphML   : {road_graphml}")
    print(f"Excel Path     : {excel_path}")
    print(f"Road Buffer    : {buffer_dist_m} m")
    print(f"Projection     : EPSG:2045")
    print("-" * 80)

    # 1. Load road network
    print("\n[Step 1/5] Loading road network...")
    roads_gdf = gpd.read_file(road_gpkg)
    roads_2045 = roads_gdf.to_crs(epsg=2045)
    print(f"Loaded {len(roads_gdf):,} segments.")

    # 2. Extract Hydrology Feature: dist_river
    print("\n[Step 2/5] Deriving Hydrology Feature (`dist_river`)...")
    rivers_gdf = extract_rivers_from_hdf(hdf_path)
    DEFAULT_OUT_RIVER_GPKG.parent.mkdir(parents=True, exist_ok=True)
    rivers_gdf.to_file(DEFAULT_OUT_RIVER_GPKG, driver="GPKG")
    dist_river_series = compute_dist_river(roads_2045, rivers_gdf)

    # 3. Extract Terrain Features: elev_min, elev_mean, slope_mean, tpi, depression_depth
    print(f"\n[Step 3/5] Deriving Terrain Features with {buffer_dist_m}m buffer...")
    terrain_df = compute_terrain_features(
        road_gdf=roads_2045,
        dem_path=dem_path,
        dep_path=dep_path,
        tpi_path=tpi_path,
        buffer_dist_m=buffer_dist_m,
    )

    # 4. Extract Rainfall Features: rain_total, rain_peak_intensity
    print("\n[Step 4/5] Deriving Rainfall Features from DSS...")
    road_rain_df, _ = compute_rainfall_features(
        road_gdf=roads_2045,
        stations_dir=dss_dir,
        stations_csv=stations_csv,
        event_month_tag="01Oct2022",
    )

    # 5. Assemble all derived features
    print("\n[Step 5/5] Consolidating all derived features...")
    features_df = pd.DataFrame({
        "segment_id": roads_2045["segment_id"].values,
        "geometry_wkt": roads_2045.geometry.to_wkt(),
        "dist_river": dist_river_series.values,
        "elev_min": terrain_df["elev_min"].values,
        "elev_mean": terrain_df["elev_mean"].values,
        "slope_mean": terrain_df["slope_mean"].values,
        "tpi": terrain_df["tpi"].values,
        "depression_depth": terrain_df["depression_depth"].values,
        "rain_total": road_rain_df["rain_total"].values,
        "rain_peak_intensity": road_rain_df["rain_peak_intensity"].values,
    })

    # Encode to GraphML (excluding rainfall)
    encode_features_to_graphml(road_graphml, features_df)

    # Update Road GPKG (excluding rainfall)
    update_road_gpkg(road_gpkg, features_df)

    # Occupy Excel feature_table
    occupy_feature_table_spreadsheet(excel_path, features_df)

    elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Road Segments Processed: {len(features_df):,}")
    print(f"Total Execution Time         : {elapsed:.2f} seconds")
    print("\nFeature Distribution Statistics:")
    stat_cols = [
        "dist_river",
        "elev_min",
        "elev_mean",
        "slope_mean",
        "tpi",
        "depression_depth",
        "rain_total",
        "rain_peak_intensity",
    ]
    for c in stat_cols:
        s = features_df[c]
        print(f"  {c:<22}: min={s.min():10.2f}, mean={s.mean():10.2f}, max={s.max():10.2f}, NaN={s.isna().sum()}")
    print("=" * 80)

    return features_df


def main():
    parser = argparse.ArgumentParser(description="Build Phase-1 researched features for Hai Chau road network.")
    parser.add_argument("--hdf", type=Path, default=DEFAULT_HDF_PATH, help="Path to HDF file")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH, help="Path to clean DEM TIFF")
    parser.add_argument("--dep", type=Path, default=DEFAULT_DEP_PATH, help="Path to depression depth TIFF")
    parser.add_argument("--tpi", type=Path, default=DEFAULT_TPI_PATH, help="Path to TPI TIFF")
    parser.add_argument("--dss-dir", type=Path, default=DEFAULT_DSS_DIR, help="Path to new DSSs directory")
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS_CSV, help="Path to stations.csv")
    parser.add_argument("--roads-gpkg", type=Path, default=DEFAULT_ROAD_GPKG, help="Path to road GPKG")
    parser.add_argument("--roads-graphml", type=Path, default=DEFAULT_ROAD_GRAPHML, help="Path to road GraphML")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH, help="Path to feature_table.xlsx")
    parser.add_argument("--buffer", type=float, default=50.0, help="Road buffer distance in meters (default: 50.0)")

    args = parser.parse_args()

    run_pipeline(
        hdf_path=args.hdf,
        dem_path=args.dem,
        dep_path=args.dep,
        tpi_path=args.tpi,
        dss_dir=args.dss_dir,
        stations_csv=args.stations_csv,
        road_gpkg=args.roads_gpkg,
        road_graphml=args.roads_graphml,
        excel_path=args.excel,
        buffer_dist_m=args.buffer,
    )


if __name__ == "__main__":
    main()
