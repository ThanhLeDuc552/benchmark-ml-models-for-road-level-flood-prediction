"""
Rainfall Feature Extraction for Hai Chau Road Network.

Derives rainfall features:
- `rain_total`: Event total precipitation assigned to each segment (mm)
- `rain_peak_intensity`: Peak hourly precipitation intensity assigned to each segment (mm/h)

Reads station data from `new DSSs/stations.csv` and HEC-DSS records from `new DSSs/<station>/precipitation.dss`.
Maps metrics to road segments via nearest-neighbor spatial join in EPSG:2045.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from pydsstools.heclib.dss import HecDss
from scipy.spatial import cKDTree

# Ensure GDAL/PROJ environment variables
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

DEFAULT_DSS_DIR = WORKSPACE_DIR / "new DSSs"
DEFAULT_STATIONS_CSV = DEFAULT_DSS_DIR / "stations.csv"
DEFAULT_ROAD_GPKG = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.gpkg"
DEFAULT_STATION_SUMMARY_CSV = SCRIPT_DIR / "station_rainfall_summary.csv"


def extract_station_rainfall_metrics(
    stations_dir: Path,
    stations_df: pd.DataFrame,
    event_month_tag: str = "01Oct2022",
) -> pd.DataFrame:
    """
    Process precipitation.dss for each station and compute:
    - rain_total (mm): Total precipitation for the event record
    - rain_peak_intensity (mm/h): Maximum 1-hour precipitation intensity
    """
    df = stations_df.copy()
    totals = []
    peaks = []

    print(f"Reading HEC-DSS precipitation records for {len(df)} stations (tag: {event_month_tag})...")
    for idx, row in df.iterrows():
        st_name = row["station_name"]
        dss_path = stations_dir / st_name / "precipitation.dss"

        total_mm = np.nan
        peak_mm_h = np.nan

        if dss_path.exists():
            try:
                with HecDss.Open(str(dss_path)) as fid:
                    path_pattern = f"/*/{st_name}/PRECIP/{event_month_tag}/1Hour/OBS/"
                    paths = fid.search_path(path_pattern)
                    if paths:
                        ts = fid.read_ts(paths[0])
                        vals = np.array(ts.values, dtype=np.float64)
                        valid = vals[vals >= 0.0]
                        if len(valid) > 0:
                            total_mm = float(valid.sum())
                            peak_mm_h = float(valid.max())
            except Exception as e:
                print(f"Warning: Failed to read DSS for {st_name} ({e})")

        totals.append(total_mm)
        peaks.append(peak_mm_h)

    df["rain_total"] = totals
    df["rain_peak_intensity"] = peaks
    return df


def assign_rainfall_to_roads(
    road_gdf: gpd.GeoDataFrame,
    stations_with_rain: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign rainfall metrics to road segments via nearest-station spatial join in EPSG:2045.
    Returns DataFrame with columns:
    ['segment_id', 'rain_total', 'rain_peak_intensity', 'nearest_station', 'dist_to_station_m']
    """
    roads_2045 = road_gdf.to_crs(epsg=2045)
    valid_stations = stations_with_rain.dropna(subset=["rain_total", "project_x_epsg_2045", "project_y_epsg_2045"])

    if valid_stations.empty:
        raise ValueError("No valid stations with precipitation data found.")

    st_coords = valid_stations[["project_x_epsg_2045", "project_y_epsg_2045"]].values
    tree = cKDTree(st_coords)

    # Road segment centroids
    road_centroids = np.array([[g.centroid.x, g.centroid.y] for g in roads_2045.geometry])

    # Nearest neighbor query
    dists, indices = tree.query(road_centroids)

    matched_stations = valid_stations.iloc[indices].reset_index(drop=True)

    result_df = pd.DataFrame({
        "segment_id": roads_2045["segment_id"].values if "segment_id" in roads_2045 else roads_2045.index,
        "rain_total": matched_stations["rain_total"].values,
        "rain_peak_intensity": matched_stations["rain_peak_intensity"].values,
        "nearest_station": matched_stations["station_name"].values,
        "dist_to_station_m": dists,
    })
    return result_df


def compute_rainfall_features(
    road_gdf: gpd.GeoDataFrame,
    stations_dir: Path = DEFAULT_DSS_DIR,
    stations_csv: Path = DEFAULT_STATIONS_CSV,
    event_month_tag: str = "01Oct2022",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline to compute road rainfall features from DSS.
    Returns:
    (road_rainfall_df, station_summary_df)
    """
    if not stations_csv.exists():
        raise FileNotFoundError(f"Stations CSV not found at: {stations_csv}")

    stations_df = pd.read_csv(stations_csv)
    stations_with_rain = extract_station_rainfall_metrics(
        stations_dir=stations_dir,
        stations_df=stations_df,
        event_month_tag=event_month_tag,
    )

    road_rainfall = assign_rainfall_to_roads(
        road_gdf=road_gdf,
        stations_with_rain=stations_with_rain,
    )

    return road_rainfall, stations_with_rain


def main():
    parser = argparse.ArgumentParser(description="Extract rainfall features from HEC-DSS records.")
    parser.add_argument("--dss-dir", type=Path, default=DEFAULT_DSS_DIR, help="Path to new DSSs directory")
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS_CSV, help="Path to stations.csv")
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROAD_GPKG, help="Path to road GPKG")
    parser.add_argument("--event-tag", type=str, default="01Oct2022", help="DSS D-part event tag (default: %(default)s)")
    parser.add_argument("--out-summary", type=Path, default=DEFAULT_STATION_SUMMARY_CSV, help="Output station summary CSV")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("RAINFALL FEATURE EXTRACTION (HEC-DSS -> Road Segments)")
    print("=" * 70)
    print(f"DSS Directory : {args.dss_dir}")
    print(f"Stations CSV  : {args.stations_csv}")
    print(f"Roads File    : {args.roads}")
    print(f"Event Tag     : {args.event_tag}")
    print(f"Output CSV    : {args.out_summary}")
    print("-" * 70)

    roads_gdf = gpd.read_file(args.roads)
    road_rain_df, stations_summary = compute_rainfall_features(
        road_gdf=roads_gdf,
        stations_dir=args.dss_dir,
        stations_csv=args.stations_csv,
        event_month_tag=args.event_tag,
    )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    stations_summary.to_csv(args.out_summary, index=False)
    print(f"Saved station rainfall summary to: {args.out_summary}")

    print("\nSummary of Derived Road Rainfall Features:")
    print(f"  rain_total (mm)          : min={road_rain_df['rain_total'].min():.2f}, mean={road_rain_df['rain_total'].mean():.2f}, max={road_rain_df['rain_total'].max():.2f}")
    print(f"  rain_peak_intensity (mm/h): min={road_rain_df['rain_peak_intensity'].min():.2f}, mean={road_rain_df['rain_peak_intensity'].mean():.2f}, max={road_rain_df['rain_peak_intensity'].max():.2f}")
    print(f"  Nearest station dist (m) : min={road_rain_df['dist_to_station_m'].min():.1f} m, mean={road_rain_df['dist_to_station_m'].mean():.1f} m, max={road_rain_df['dist_to_station_m'].max():.1f} m")
    print(f"  Unique stations matched  : {road_rain_df['nearest_station'].nunique()}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
