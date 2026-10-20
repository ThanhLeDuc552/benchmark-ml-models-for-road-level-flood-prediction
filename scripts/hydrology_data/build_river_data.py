"""
Hydrology Feature Extraction for Hai Chau Road Network.

Derives `dist_river` (distance from road segments to nearest major river)
using river geometries extracted from HEC-RAS geometry in `danangfloodmodel.g01.hdf`.

CRS: EPSG:2045 (Hanoi 1972 / GK Zone 19).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

# Ensure standard GDAL/PROJ environment variables
conda_prefix = Path(sys.executable).parent
gdal_data = conda_prefix / "Library" / "share" / "gdal"
proj_lib = conda_prefix / "Library" / "share" / "proj"
if gdal_data.exists():
    os.environ["GDAL_DATA"] = str(gdal_data)
if proj_lib.exists():
    os.environ["PROJ_LIB"] = str(proj_lib)
    os.environ["PROJ_DATA"] = str(proj_lib)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent
DEFAULT_HDF_PATH = WORKSPACE_DIR / "resources for forming dataset" / "danangfloodmodel.g01.hdf"
DEFAULT_ROAD_GPKG = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.gpkg"
DEFAULT_OUT_RIVER_GPKG = SCRIPT_DIR / "danang_rivers_epsg2045.gpkg"


def extract_rivers_from_hdf(hdf_path: Path) -> gpd.GeoDataFrame:
    """
    Extract all river centerlines from HEC-RAS HDF geometry.
    Coordinates in the HDF file are in Hanoi 1972 / GK Zone 19 (EPSG:2045).
    """
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF file not found: {hdf_path}")

    records = []
    with h5py.File(hdf_path, "r") as f:
        rc_g = f["Geometry/River Centerlines"]
        info = rc_g["Polyline Info"][:]
        pts = rc_g["Polyline Points"][:]
        attrs = rc_g["Attributes"][:]

        for i in range(len(info)):
            r_name = attrs[i][0].decode("utf-8", errors="replace").strip()
            reach_name = attrs[i][1].decode("utf-8", errors="replace").strip()
            start_idx = info[i, 0]
            num_pts = info[i, 1]
            sub_pts = pts[start_idx : start_idx + num_pts]
            line = LineString(sub_pts)

            records.append({
                "river_index": i,
                "river_name": r_name,
                "reach_name": reach_name,
                "point_count": num_pts,
                "length_m": line.length,
                "geometry": line,
            })

    gdf_rivers = gpd.GeoDataFrame(records, crs="EPSG:2045")
    return gdf_rivers


def compute_dist_river(
    road_gdf: gpd.GeoDataFrame,
    river_gdf: gpd.GeoDataFrame,
    target_river: str | None = None,
) -> pd.Series:
    """
    Compute Euclidean distance (in meters) from each road segment to the river network in EPSG:2045.
    If target_river is specified (e.g. 'Han'), measures distance to that specific river.
    Otherwise measures distance to the entire river network.
    """
    roads_2045 = road_gdf.to_crs(epsg=2045)

    if target_river:
        filtered = river_gdf[river_gdf["river_name"].str.lower() == target_river.lower()]
        if filtered.empty:
            raise ValueError(f"River '{target_river}' not found in river GeoDataFrame.")
        river_geom = filtered.geometry.union_all()
    else:
        river_geom = river_gdf.geometry.union_all()

    # Euclidean distance from segment LineString to river geometry
    distances = roads_2045.geometry.apply(lambda g: float(g.distance(river_geom)))
    return distances


def main():
    parser = argparse.ArgumentParser(description="Extract river geometry and compute dist_river.")
    parser.add_argument("--hdf", type=Path, default=DEFAULT_HDF_PATH, help="Path to HDF file")
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROAD_GPKG, help="Path to road gpkg")
    parser.add_argument("--out-rivers", type=Path, default=DEFAULT_OUT_RIVER_GPKG, help="Output river GPKG")
    args = parser.parse_args()

    print("=" * 70)
    print("HYDROLOGY FEATURE EXTRACTION: dist_river")
    print("=" * 70)
    print(f"HDF File    : {args.hdf}")
    print(f"Roads File  : {args.roads}")
    print(f"Rivers Out  : {args.out_rivers}")
    print("-" * 70)

    print("1. Extracting river centerlines from HDF...")
    rivers_gdf = extract_rivers_from_hdf(args.hdf)
    print(f"Extracted {len(rivers_gdf)} river centerlines:")
    for _, row in rivers_gdf.iterrows():
        print(f"  - {row['river_name']} ({row['reach_name']}): {row['length_m']:.1f} m")

    args.out_rivers.parent.mkdir(parents=True, exist_ok=True)
    rivers_gdf.to_file(args.out_rivers, driver="GPKG")
    print(f"Saved river network vector to: {args.out_rivers}")

    print("\n2. Computing dist_river for road segments...")
    roads_gdf = gpd.read_file(args.roads)
    dist_river_series = compute_dist_river(roads_gdf, rivers_gdf)

    print("\nSummary of dist_river (meters):")
    print(f"  Count : {len(dist_river_series):,}")
    print(f"  Min   : {dist_river_series.min():.2f} m")
    print(f"  Mean  : {dist_river_series.mean():.2f} m")
    print(f"  Max   : {dist_river_series.max():.2f} m")
    print("=" * 70)


if __name__ == "__main__":
    main()
