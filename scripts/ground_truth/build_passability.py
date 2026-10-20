"""
Ground Truth Feature Extraction: Road Passability Classification.

Derives the `passability` feature based on the HEC-RAS 2D unsteady hydraulic simulation run
(`2022-10-14_to_2022-10-15/danangfloodmodel.p05.hdf` / `water_surface/combined.npy`)
and the segment geometry `geometry_wkt` (EPSG:2045).

Categories (water depth during flood event):
- 0: passable            (0 - 10 cm, depth < 0.10 m)
- 1: flooded - passable   (10 - 30 cm, 0.10 m <= depth <= 0.30 m)
- 2: non-passable        (> 30 cm, depth > 0.30 m)

Updates:
1. `feature_table.xlsx`: sheet `feature_table`, column `passability` (dtype: 0/1/2)
2. `hai_chau_road_graph.graphml`: edge attribute `passability`
3. `hai_chau_road_graph.gpkg`: column `passability`
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
import h5py
import networkx as nx
import numpy as np
import openpyxl
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkt

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent

DEFAULT_EVENT_DIR = WORKSPACE_DIR / "resources for forming dataset" / "hec_ras_run_2022-10-14_to_2022-10-15"
DEFAULT_P05_PATH = DEFAULT_EVENT_DIR / "danangfloodmodel.p05.hdf"
DEFAULT_EXCEL_PATH = WORKSPACE_DIR / "dataset" / "feature_table.xlsx"
DEFAULT_GRAPHML_PATH = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.graphml"
DEFAULT_GPKG_PATH = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.gpkg"
DEFAULT_OUT_CSV = SCRIPT_DIR / "passability_summary.csv"


def extract_hecras_mesh_peak_depths(p05_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract 2D mesh cell center coordinates and event peak inundation depths (m)
    from HEC-RAS Plan 05 HDF file across Perimeter 1 and Perimeter 2.
    """
    if not p05_path.exists():
        raise FileNotFoundError(f"HEC-RAS HDF file not found: {p05_path}")

    all_coords = []
    all_peak_depths = []

    with h5py.File(p05_path, "r") as f:
        ts_2d = f["Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas"]
        for p in ["Perimeter 1", "Perimeter 2"]:
            if p not in ts_2d:
                continue
            coords = f[f"Geometry/2D Flow Areas/{p}/Cells Center Coordinate"][:]
            elev = f[f"Geometry/2D Flow Areas/{p}/Cells Minimum Elevation"][:]
            wse = ts_2d[p]["Water Surface"][:]  # shape: (T, N_cells)

            # Peak water surface elevation across all timesteps
            max_wse = np.nanmax(wse, axis=0)

            # Inundation depth: max(0, WSE - Cell Elevation)
            depth = np.maximum(0.0, max_wse - elev)
            valid = ~np.isnan(depth) & ~np.isnan(elev) & (elev > -100.0)

            all_coords.append(coords[valid])
            all_peak_depths.append(depth[valid].astype(np.float32))

    cells_coords = np.vstack(all_coords)
    cells_depth = np.concatenate(all_peak_depths)
    return cells_coords, cells_depth


def compute_road_passability(
    geometries: list,
    cells_coords: np.ndarray,
    cells_depth: np.ndarray,
    sample_spacing_m: float = 10.0,
    depth_stat: str = "max",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample peak water depth along each road segment geometry (EPSG:2045) and
    classify into passability category (0, 1, 2).

    Parameters:
    - geometries: list of Shapely LineString geometries (EPSG:2045)
    - cells_coords: (N, 2) array of HEC-RAS cell center coordinates (EPSG:2045)
    - cells_depth: (N,) array of peak inundation depths (meters)
    - sample_spacing_m: distance in meters between sample points along segment
    - depth_stat: 'max' (critical bottleneck depth) or 'mean' (segment average)

    Returns:
    - road_depths: (len(geometries),) peak depth in meters
    - passability_labels: (len(geometries),) integer categories 0, 1, 2
    """
    tree = cKDTree(cells_coords)
    road_depths = []

    for g in geometries:
        length = g.length
        # Sample points along line at sample_spacing_m spacing (at least 3 points: start, mid, end)
        num_pts = max(3, int(np.ceil(length / sample_spacing_m)) + 1)
        fractions = np.linspace(0.0, 1.0, num_pts)
        pts = [g.interpolate(f, normalized=True) for f in fractions]
        pts_coords = np.array([[p.x, p.y] for p in pts])

        # Query nearest cell for each point along the segment
        _, idxs = tree.query(pts_coords, k=1)
        sampled_depths = cells_depth[idxs]

        if depth_stat == "mean":
            d_val = float(sampled_depths.mean())
        else:  # default 'max' (critical bottleneck)
            d_val = float(sampled_depths.max())

        road_depths.append(d_val)

    road_depths = np.array(road_depths, dtype=np.float32)

    # Classify according to user specifications:
    # - passable: 0 - 10 cm (depth < 0.10 m)        -> 0
    # - flooded - passable: 10 - 30 cm (0.10 - 0.30 m) -> 1
    # - non-passable: > 30 cm (depth > 0.30 m)         -> 2
    labels = np.zeros(len(road_depths), dtype=int)
    labels[(road_depths >= 0.10) & (road_depths <= 0.30)] = 1
    labels[road_depths > 0.30] = 2

    return road_depths, labels


def update_excel_passability(excel_path: Path, segment_ids: list[str], labels: np.ndarray) -> None:
    """
    Update the 'passability' column in 'feature_table' sheet of feature_table.xlsx.
    """
    print(f"\nOpening Excel workbook: {excel_path.name} ...")
    wb = openpyxl.load_workbook(excel_path)
    if "feature_table" not in wb.sheetnames:
        raise ValueError(f"'feature_table' sheet missing from {excel_path}")

    ws = wb["feature_table"]
    header_row = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    col_map = {name: idx + 1 for idx, name in enumerate(header_row) if name is not None}

    if "passability" not in col_map:
        raise KeyError(f"'passability' column not found in headers: {header_row}")
    if "segment_id" not in col_map:
        raise KeyError(f"'segment_id' column not found in headers: {header_row}")

    pass_col_idx = col_map["passability"]
    seg_col_idx = col_map["segment_id"]

    label_lookup = dict(zip(segment_ids, [int(x) for x in labels]))

    updated = 0
    for r in range(2, ws.max_row + 1):
        seg_id = ws.cell(r, seg_col_idx).value
        if seg_id in label_lookup:
            ws.cell(r, pass_col_idx).value = label_lookup[seg_id]
            updated += 1

    wb.save(excel_path)
    print(f"Successfully updated 'passability' column for {updated:,} rows in {excel_path.name}.")


def update_graphml_passability(graphml_path: Path, segment_ids: list[str], labels: np.ndarray) -> None:
    """
    Encode 'passability' edge attribute into hai_chau_road_graph.graphml.
    """
    print(f"\nUpdating GraphML: {graphml_path.name} ...")
    G = nx.read_graphml(graphml_path)
    label_lookup = dict(zip(segment_ids, [int(x) for x in labels]))

    updated = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        seg_id = d.get("segment_id")
        if seg_id in label_lookup:
            d["passability"] = label_lookup[seg_id]
            updated += 1

    nx.write_graphml(G, graphml_path)
    print(f"Encoded 'passability' attribute into {updated:,} edges in {graphml_path.name}.")


def update_gpkg_passability(gpkg_path: Path, segment_ids: list[str], labels: np.ndarray) -> None:
    """
    Update 'passability' column in hai_chau_road_graph.gpkg.
    """
    print(f"\nUpdating road GeoPackage: {gpkg_path.name} ...")
    gdf = gpd.read_file(gpkg_path)
    label_df = pd.DataFrame({"segment_id": segment_ids, "passability": labels})

    if "passability" in gdf.columns:
        gdf = gdf.drop(columns=["passability"])

    merged = gdf.merge(label_df, on="segment_id", how="left")
    merged.to_file(gpkg_path, driver="GPKG")
    print(f"Saved updated GeoPackage to {gpkg_path.name}.")


def main():
    parser = argparse.ArgumentParser(description="Derive road passability feature from HEC-RAS run.")
    parser.add_argument("--p05", type=Path, default=DEFAULT_P05_PATH, help="Path to danangfloodmodel.p05.hdf")
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL_PATH, help="Path to feature_table.xlsx")
    parser.add_argument("--graphml", type=Path, default=DEFAULT_GRAPHML_PATH, help="Path to GraphML")
    parser.add_argument("--gpkg", type=Path, default=DEFAULT_GPKG_PATH, help="Path to GPKG")
    parser.add_argument("--depth-stat", type=str, default="max", choices=["max", "mean"], help="Depth stat along line (default: max)")
    parser.add_argument("--spacing", type=float, default=10.0, help="Sampling spacing along line in meters (default: 10.0)")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV, help="Path to output summary CSV")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 75)
    print("GROUND TRUTH FEATURE EXTRACTION: Road Passability (HEC-RAS Run)")
    print("=" * 75)
    print(f"HEC-RAS Run : {args.p05}")
    print(f"Excel File  : {args.excel}")
    print(f"GraphML File: {args.graphml}")
    print(f"GPKG File   : {args.gpkg}")
    print(f"Depth Stat  : {args.depth_stat} (bottleneck sampling every {args.spacing} m)")
    print("-" * 75)

    # 1. Load road segments and geometry_wkt
    print("\n[Step 1/4] Loading road segment geometries from feature_table.xlsx...")
    ft = pd.read_excel(args.excel, sheet_name="feature_table")
    if "geometry_wkt" not in ft.columns:
        raise KeyError("'geometry_wkt' column missing from feature_table sheet.")

    segment_ids = ft["segment_id"].tolist()
    geometries = [wkt.loads(s) for s in ft["geometry_wkt"]]
    print(f"Loaded {len(geometries):,} road segments (CRS: EPSG:2045).")

    # 2. Extract 2D cell depths from HEC-RAS Plan 05 HDF
    print("\n[Step 2/4] Extracting 2D mesh peak flood depths from HEC-RAS run...")
    cells_coords, cells_depth = extract_hecras_mesh_peak_depths(args.p05)
    print(f"Extracted {len(cells_coords):,} 2D mesh cells:")
    print(f"  Peak event depth range: {cells_depth.min():.2f} m to {cells_depth.max():.2f} m (mean: {cells_depth.mean():.2f} m)")

    # 3. Compute road passability
    print(f"\n[Step 3/4] Sampling flood depth along road lines (spacing: {args.spacing}m, stat: {args.depth_stat})...")
    road_depths, labels = compute_road_passability(
        geometries=geometries,
        cells_coords=cells_coords,
        cells_depth=cells_depth,
        sample_spacing_m=args.spacing,
        depth_stat=args.depth_stat,
    )

    # Classification distribution
    count_0 = int((labels == 0).sum())
    count_1 = int((labels == 1).sum())
    count_2 = int((labels == 2).sum())
    total = len(labels)

    print("\n" + "=" * 75)
    print("PASSABILITY CLASSIFICATION RESULTS:")
    print("=" * 75)
    print(f"  [0] Passable            (0 - 10 cm, depth < 0.10 m)  : {count_0:5,} ({count_0/total*100:5.1f}%)")
    print(f"  [1] Flooded - Passable  (10 - 30 cm, 0.10 - 0.30 m) : {count_1:5,} ({count_1/total*100:5.1f}%)")
    print(f"  [2] Non-passable        (> 30 cm, depth > 0.30 m)    : {count_2:5,} ({count_2/total*100:5.1f}%)")
    print(f"  Total Segments Classified                            : {total:5,} (100.0%)")
    print(f"  Sampled Road Depth Range                             : {road_depths.min():.3f} m to {road_depths.max():.3f} m (mean: {road_depths.mean():.3f} m)")
    print("=" * 75)

    # 4. Update targets in files
    print("\n[Step 4/4] Writing passability to datasets...")
    update_excel_passability(args.excel, segment_ids, labels)
    update_graphml_passability(args.graphml, segment_ids, labels)
    update_gpkg_passability(args.gpkg, segment_ids, labels)

    # Save summary CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame({
        "segment_id": segment_ids,
        "peak_water_depth_m": road_depths,
        "passability": labels,
        "category_name": [
            "passable" if l == 0 else ("flooded - passable" if l == 1 else "non-passable")
            for l in labels
        ],
    })
    summary_df.to_csv(args.out_csv, index=False)
    print(f"Saved passability summary CSV to: {args.out_csv}")

    elapsed = time.time() - t0
    print(f"\nAll operations completed successfully in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
