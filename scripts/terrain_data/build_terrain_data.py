"""
Terrain Feature Extraction for Hai Chau Road Network.

Derives terrain features:
- `elev_min`: Minimum elevation within 50m buffer (meters)
- `elev_mean`: Mean elevation within 50m buffer (meters)
- `slope_mean`: Mean slope within 50m buffer (degrees), computed via Horn's method
- `tpi`: Mean Topographic Position Index within 50m buffer (meters)
- `depression_depth`: Maximum depression depth within 50m buffer (meters)

All computations are performed in project projection EPSG:2045 (Hanoi 1972 / GK Zone 19).
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
import rasterio
from rasterio.features import geometry_mask
from scipy.ndimage import convolve

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

DEFAULT_DEM_PATH = WORKSPACE_DIR / "resources for forming dataset" / "COPERNICUS_DANANG_FULL_EPSG2045_clean.tif"
DEFAULT_DEP_PATH = WORKSPACE_DIR / "resources for forming dataset" / "depression_depth.tif"
DEFAULT_TPI_PATH = WORKSPACE_DIR / "resources for forming dataset" / "tpi.tif"
DEFAULT_ROAD_GPKG = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.gpkg"
DEFAULT_OUT_CSV = SCRIPT_DIR / "terrain_features_cache.csv"


def compute_horn_slope(dem: np.ndarray, res_x: float, res_y: float, nodata: float) -> np.ndarray:
    """
    Compute slope in degrees using Horn's (1981) 3x3 finite-difference algorithm,
    identical to GDAL `gdaldem slope -s 1.0`.
    """
    valid_mask = (dem != nodata) & (~np.isnan(dem)) & (dem > -100)
    dem_filled = np.where(valid_mask, dem, 0.0)

    # Horn kernels
    kx = np.array([[-1.0, 0.0, 1.0],
                   [-2.0, 0.0, 2.0],
                   [-1.0, 0.0, 1.0]]) / (8.0 * res_x)

    ky = np.array([[ 1.0,  2.0,  1.0],
                   [ 0.0,  0.0,  0.0],
                   [-1.0, -2.0, -1.0]]) / (8.0 * res_y)

    dz_dx = convolve(dem_filled, kx, mode="nearest")
    dz_dy = convolve(dem_filled, ky, mode="nearest")

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    slope_deg[~valid_mask] = np.nan
    return slope_deg.astype(np.float32)


def extract_zonal_stats_for_buffers(
    raster_arr: np.ndarray,
    trans: rasterio.Affine,
    nodata: float,
    buffers: list,
    stat: str = "mean",
    min_valid: float = -1e30,
) -> list[float]:
    """
    Perform fast windowed zonal extraction for a list of polygon buffers.
    `stat` can be 'mean', 'min', or 'max'.
    """
    results = []
    h, w = raster_arr.shape

    for poly in buffers:
        minx, miny, maxx, maxy = poly.bounds
        r_max, c_min = rasterio.transform.rowcol(trans, minx, miny)
        r_min, c_max = rasterio.transform.rowcol(trans, maxx, maxy)

        r_start = max(0, min(r_min, r_max) - 1)
        r_end = min(h, max(r_min, r_max) + 2)
        c_start = max(0, min(c_min, c_max) - 1)
        c_end = min(w, max(c_min, c_max) + 2)

        sub_arr = raster_arr[r_start:r_end, c_start:c_end]
        sub_trans = rasterio.windows.transform(
            rasterio.windows.Window(c_start, r_start, c_end - c_start, r_end - r_start),
            trans,
        )

        mask = geometry_mask(
            [poly],
            transform=sub_trans,
            invert=True,
            out_shape=sub_arr.shape,
            all_touched=True,
        )

        cells = sub_arr[mask]
        valid_cells = cells[
            (cells != nodata) & (~np.isnan(cells)) & (cells > min_valid)
        ]

        if len(valid_cells) > 0:
            if stat == "min":
                val = float(valid_cells.min())
            elif stat == "max":
                val = float(valid_cells.max())
            else:  # mean
                val = float(valid_cells.mean())
        else:
            val = np.nan
        results.append(val)

    return results


def compute_terrain_features(
    road_gdf: gpd.GeoDataFrame,
    dem_path: Path = DEFAULT_DEM_PATH,
    dep_path: Path = DEFAULT_DEP_PATH,
    tpi_path: Path = DEFAULT_TPI_PATH,
    buffer_dist_m: float = 50.0,
) -> pd.DataFrame:
    """
    Compute all phase-1 researched terrain features for the road network.
    Returns a DataFrame with columns:
    ['segment_id', 'elev_min', 'elev_mean', 'slope_mean', 'tpi', 'depression_depth']
    """
    print(f"Reprojecting roads to EPSG:2045 and generating {buffer_dist_m}m buffers...")
    roads_2045 = road_gdf.to_crs(epsg=2045)
    buffers = roads_2045.geometry.buffer(buffer_dist_m).tolist()

    # 1. Load DEM & compute Horn slope
    print(f"Loading DEM: {dem_path.name} ...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        dem_trans = src.transform
        dem_nodata = src.nodata if src.nodata is not None else -9999.0
        res_x, res_y = src.res

    print("Computing Horn slope (degrees)...")
    slope = compute_horn_slope(dem, res_x, res_y, dem_nodata)

    # 2. Extract elev_min, elev_mean, slope_mean
    print("Computing zonal elevation (min & mean)...")
    elev_min = extract_zonal_stats_for_buffers(
        dem, dem_trans, dem_nodata, buffers, stat="min", min_valid=-100.0
    )
    elev_mean = extract_zonal_stats_for_buffers(
        dem, dem_trans, dem_nodata, buffers, stat="mean", min_valid=-100.0
    )

    print("Computing zonal slope (mean)...")
    slope_mean = extract_zonal_stats_for_buffers(
        slope, dem_trans, np.nan, buffers, stat="mean", min_valid=-10.0
    )

    # 3. Load TPI and extract mean
    print(f"Loading TPI: {tpi_path.name} ...")
    with rasterio.open(tpi_path) as src:
        tpi_arr = src.read(1)
        tpi_trans = src.transform
        tpi_nodata = src.nodata if src.nodata is not None else -9999.0

    print("Computing zonal TPI (mean)...")
    tpi_mean = extract_zonal_stats_for_buffers(
        tpi_arr, tpi_trans, tpi_nodata, buffers, stat="mean", min_valid=-1e20
    )

    # 4. Load Depression Depth and extract max
    print(f"Loading Depression Depth: {dep_path.name} ...")
    with rasterio.open(dep_path) as src:
        dep_arr = src.read(1)
        dep_trans = src.transform
        dep_nodata = src.nodata if src.nodata is not None else -3.4028235e38

    print("Computing zonal depression_depth (max)...")
    dep_max = extract_zonal_stats_for_buffers(
        dep_arr, dep_trans, dep_nodata, buffers, stat="max", min_valid=-1e20
    )

    df_out = pd.DataFrame({
        "segment_id": roads_2045["segment_id"].values if "segment_id" in roads_2045 else roads_2045.index,
        "elev_min": elev_min,
        "elev_mean": elev_mean,
        "slope_mean": slope_mean,
        "tpi": tpi_mean,
        "depression_depth": dep_max,
    })
    return df_out


def main():
    parser = argparse.ArgumentParser(description="Extract terrain features for road network buffers.")
    parser.add_argument("--dem", type=Path, default=DEFAULT_DEM_PATH, help="Path to clean DEM TIFF")
    parser.add_argument("--dep", type=Path, default=DEFAULT_DEP_PATH, help="Path to depression depth TIFF")
    parser.add_argument("--tpi", type=Path, default=DEFAULT_TPI_PATH, help="Path to TPI TIFF")
    parser.add_argument("--roads", type=Path, default=DEFAULT_ROAD_GPKG, help="Path to road GPKG")
    parser.add_argument("--buffer", type=float, default=50.0, help="Road buffer distance in meters")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Output cache CSV")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 70)
    print("TERRAIN FEATURE EXTRACTION (50m Buffer)")
    print("=" * 70)
    print(f"DEM File     : {args.dem}")
    print(f"Depression   : {args.dep}")
    print(f"TPI File     : {args.tpi}")
    print(f"Roads File   : {args.roads}")
    print(f"Buffer Dist  : {args.buffer} m")
    print(f"Output CSV   : {args.out_csv}")
    print("-" * 70)

    roads_gdf = gpd.read_file(args.roads)
    terrain_df = compute_terrain_features(
        road_gdf=roads_gdf,
        dem_path=args.dem,
        dep_path=args.dep,
        tpi_path=args.tpi,
        buffer_dist_m=args.buffer,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    terrain_df.to_csv(args.out_csv, index=False)
    print(f"\nSaved terrain features to: {args.out_csv}")

    print("\nSummary of Terrain Features:")
    for col in ["elev_min", "elev_mean", "slope_mean", "tpi", "depression_depth"]:
        s = terrain_df[col]
        print(f"  {col:<17}: min={s.min():.2f}, mean={s.mean():.2f}, max={s.max():.2f}, nan_count={s.isna().sum()}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
