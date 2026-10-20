"""
Extract road and infrastructure spatial data for Da Nang from Vietnam GeoPackage.

This script extracts features from Vietnam OSM GeoPackage (vietnam.gpkg)
filtered/clipped by the administrative boundary in Da Nang GeoJSON:
`Data Collections/Đà Nẵng (tỉnh thành) - 34_simplified.geojson`.

Extracted layers are written to a GeoPackage placed in the same directory
as the source dataset:
`Data Collections/extras/vietnam-261008-free.gpkg/danang_roads.gpkg`

Features:
- Fast spatial index filtering using pyogrio / geopandas.
- Supports flexible layer extraction:
    * Default: 'gis_osm_roads_free' (primary road network)
    * Road & Transit suite: '--all-road-layers' (roads, railways, traffic, transit)
    * Full database extract: '--all-layers' (all 20 OSM layers for Da Nang)
    * Custom layers: '--layers layer1,layer2'
- Spatial filtering modes:
    * 'intersect' (default): keeps entire geometry segments intersecting Da Nang.
    * 'exact': clips line/polygon geometries precisely along the boundary perimeter.
    * 'bbox': fast rectangular bounding box filtering.
- Automatic CRS alignment and validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional

# Ensure UTF-8 output encoding for Windows PowerShell / cmd consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import pyogrio


# ============================================================
# DEFAULT PATH RESOLUTION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_GPKG_PATH = (
    PROJECT_ROOT / "Data Collections" / "extras" / "vietnam-261008-free.gpkg" / "vietnam.gpkg"
)
DEFAULT_BOUNDARY_PATH = (
    PROJECT_ROOT / "Data Collections" / "Đà Nẵng (tỉnh thành) - 34_simplified.geojson"
)
DEFAULT_OUTPUT_GPKG = (
    PROJECT_ROOT / "Data Collections" / "extras" / "vietnam-261008-free.gpkg" / "danang_roads.gpkg"
)

# Standard road & transport related layers
ROAD_TRANSIT_LAYERS = [
    "gis_osm_roads_free",
    "gis_osm_railways_free",
    "gis_osm_traffic_free",
    "gis_osm_traffic_a_free",
    "gis_osm_transport_free",
    "gis_osm_transport_a_free",
]

# Major highway, trunk, primary, secondary corridors and link roads
DEFAULT_INCLUDED_ROAD_TYPES = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def list_gpkg_layers(gpkg_path: Path) -> List[str]:
    """Return all available layer names in a GeoPackage."""
    layers_info = pyogrio.list_layers(gpkg_path)
    return [layer[0] for layer in layers_info]


def load_boundary_polygon(geojson_path: Path, target_crs=None) -> gpd.GeoDataFrame:
    """Load administrative boundary GeoJSON and reproject if needed."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"Boundary GeoJSON not found: {geojson_path}")

    boundary_gdf = gpd.read_file(geojson_path)
    if boundary_gdf.empty:
        raise ValueError(f"Boundary GeoJSON contains no features: {geojson_path}")

    if target_crs is not None and boundary_gdf.crs != target_crs:
        boundary_gdf = boundary_gdf.to_crs(target_crs)

    return boundary_gdf


def extract_layer(
    gpkg_path: Path,
    layer_name: str,
    boundary_gdf: gpd.GeoDataFrame,
    clip_mode: str = "intersect",
) -> gpd.GeoDataFrame:
    """
    Extract a single layer from GeoPackage within boundary polygon.

    Parameters:
    -----------
    gpkg_path : Path
        Source GeoPackage file path.
    layer_name : str
        Name of layer to extract.
    boundary_gdf : GeoDataFrame
        Boundary polygon in matching CRS.
    clip_mode : str
        'intersect' (filter intersecting features),
        'exact' (geometrically clip features to boundary),
        'bbox' (filter by bounding box only).
    """
    polygon = boundary_gdf.geometry.union_all()
    bbox = tuple(boundary_gdf.total_bounds)

    # 1. Fast read using pyogrio with spatial index pre-filter
    if clip_mode == "bbox":
        gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name, bbox=bbox)
        return gdf

    # For intersect or exact, first query with mask or bbox
    # Using mask parameter in pyogrio uses spatial index directly
    try:
        gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name, mask=polygon)
    except Exception:
        # Fallback to bbox read then manual spatial filter
        gdf = pyogrio.read_dataframe(gpkg_path, layer=layer_name, bbox=bbox)
        if gdf.crs != boundary_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(gdf.crs)
            polygon = boundary_gdf.geometry.union_all()
        gdf = gdf[gdf.geometry.intersects(polygon)].copy()

    # 2. Perform exact geometric clipping if requested
    if clip_mode == "exact":
        if gdf.crs != boundary_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(gdf.crs)
        gdf = gpd.clip(gdf, boundary_gdf)

    return gdf


def run_extraction(
    gpkg_path: Path = DEFAULT_GPKG_PATH,
    boundary_path: Path = DEFAULT_BOUNDARY_PATH,
    output_path: Path = DEFAULT_OUTPUT_GPKG,
    layers: Optional[List[str]] = None,
    road_types: Optional[List[str]] = None,
    all_road_types: bool = False,
    all_layers: bool = False,
    all_road_layers: bool = False,
    clip_mode: str = "intersect",
    overwrite: bool = True,
) -> Path:
    """
    Execute spatial extraction workflow from Vietnam GPKG to Da Nang GPKG.
    """
    gpkg_path = Path(gpkg_path)
    boundary_path = Path(boundary_path)
    output_path = Path(output_path)

    print("=" * 70)
    print("DA NANG OSM DATA EXTRACTION")
    print("=" * 70)
    print(f"Source GPKG     : {gpkg_path}")
    print(f"Boundary File   : {boundary_path}")
    print(f"Output GPKG     : {output_path}")
    print(f"Clipping Mode   : {clip_mode}")

    if not gpkg_path.exists():
        raise FileNotFoundError(f"Source GeoPackage not found: {gpkg_path}")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_path}")

    # Inspect source GPKG layers
    available_layers = list_gpkg_layers(gpkg_path)
    print(f"Source layers available: {len(available_layers)}")

    # Determine layers to extract
    if all_layers:
        selected_layers = available_layers
    elif all_road_layers:
        selected_layers = [l for l in ROAD_TRANSIT_LAYERS if l in available_layers]
    elif layers:
        selected_layers = [l.strip() for l in layers if l.strip() in available_layers]
        missing = [l.strip() for l in layers if l.strip() not in available_layers]
        if missing:
            print(f"Warning: Layers not found in source: {missing}")
    else:
        # Default: roads layer
        selected_layers = ["gis_osm_roads_free"]

    if not selected_layers:
        raise ValueError("No valid layers selected for extraction.")

    print(f"Selected layers ({len(selected_layers)}): {selected_layers}")
    if road_types:
        print(f"Filtering roads to specific types ({len(road_types)}): {road_types}")
    elif not all_road_types:
        print(f"Filtering roads to major corridor types ({len(DEFAULT_INCLUDED_ROAD_TYPES)}): {DEFAULT_INCLUDED_ROAD_TYPES}")
    else:
        print("Extracting all road types (unfiltered).")
    print("-" * 70)

    # Load Da Nang boundary
    # Vietnam GPKG typically uses WGS84 (EPSG:4326 / CRS84)
    boundary_gdf = load_boundary_polygon(boundary_path)
    print(f"Boundary CRS    : {boundary_gdf.crs}")
    print(f"Boundary bounds : {boundary_gdf.total_bounds.tolist()}")
    print("-" * 70)

    # Prepare output destination directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f"{output_path.stem}_tmp{output_path.suffix}")
    if temp_output_path.exists():
        temp_output_path.unlink()

    total_start = time.time()
    total_features = 0

    for idx, layer_name in enumerate(selected_layers, start=1):
        t0 = time.time()
        print(f"[{idx}/{len(selected_layers)}] Extracting layer: '{layer_name}' ...", end="", flush=True)

        try:
            # Match boundary CRS with layer if known
            layer_info = pyogrio.read_info(gpkg_path, layer=layer_name)
            layer_crs = layer_info.get("crs")
            if layer_crs and boundary_gdf.crs != layer_crs:
                boundary_layer = boundary_gdf.to_crs(layer_crs)
            else:
                boundary_layer = boundary_gdf

            extracted_gdf = extract_layer(
                gpkg_path=gpkg_path,
                layer_name=layer_name,
                boundary_gdf=boundary_layer,
                clip_mode=clip_mode,
            )

            # Apply road type filtering if applicable
            if layer_name == "gis_osm_roads_free":
                before_count = len(extracted_gdf)
                if road_types:
                    extracted_gdf = extracted_gdf[extracted_gdf["fclass"].isin(road_types)].copy()
                    print(f" [Filtered {len(extracted_gdf):,}/{before_count:,} roads matching --road-types]", end="")
                elif not all_road_types:
                    extracted_gdf = extracted_gdf[extracted_gdf["fclass"].isin(DEFAULT_INCLUDED_ROAD_TYPES)].copy()
                    print(f" [Filtered {len(extracted_gdf):,}/{before_count:,} major corridor roads]", end="")

            count = len(extracted_gdf)
            elapsed = time.time() - t0
            print(f" Done! Extracted {count:,} features in {elapsed:.2f}s")

            if count > 0:
                # Write to temp output GPKG
                write_mode = "a" if temp_output_path.exists() else "w"
                extracted_gdf.to_file(
                    temp_output_path,
                    layer=layer_name,
                    driver="GPKG",
                    mode=write_mode,
                    engine="pyogrio",
                )
                total_features += count
            else:
                print(f"   (Skipping empty layer write for '{layer_name}')")

        except Exception as e:
            print(f"\n   Error extracting '{layer_name}': {e}")

    # Safely transfer temp file to final output_path (handles Windows file locks)
    if temp_output_path.exists():
        with open(temp_output_path, "rb") as fin:
            data = fin.read()
        with open(output_path, "wb") as fout:
            fout.write(data)
        temp_output_path.unlink()

    total_time = time.time() - total_start
    output_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0

    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Output File     : {output_path}")
    print(f"Output File Size: {output_size_mb:.2f} MB")
    print(f"Layers Written  : {len(selected_layers)}")
    print(f"Total Features  : {total_features:,}")
    print(f"Total Time      : {total_time:.2f} seconds")
    print("=" * 70)

    return output_path


# ============================================================
# CLI ENTRYPOINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract Da Nang road and infrastructure data from Vietnam OSM GeoPackage."
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_GPKG_PATH,
        help="Path to source Vietnam GeoPackage (default: %(default)s)",
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=DEFAULT_BOUNDARY_PATH,
        help="Path to Da Nang boundary GeoJSON (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_GPKG,
        help="Path to output GeoPackage (default: %(default)s)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated list of layers to extract (e.g. 'gis_osm_roads_free,gis_osm_traffic_free')",
    )
    parser.add_argument(
        "--road-types",
        type=str,
        default=None,
        help="Comma-separated list of specific road types to extract (default: motorway,trunk,primary,secondary,and links)",
    )
    parser.add_argument(
        "--all-road-types",
        action="store_true",
        help="Include all OSM road types without filtering to default major corridors",
    )
    parser.add_argument(
        "--all-road-layers",
        action="store_true",
        help="Extract all road and transit layers (roads, railways, traffic, transport)",
    )
    parser.add_argument(
        "--all-layers",
        action="store_true",
        help="Extract all available layers from the source GeoPackage",
    )
    parser.add_argument(
        "--clip-mode",
        choices=["intersect", "exact", "bbox"],
        default="intersect",
        help="Spatial filtering mode: 'intersect' (default: keeps intersecting features), "
             "'exact' (clips lines/polygons cleanly at boundary), 'bbox' (bounding box)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output file",
    )
    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="List all layers present in the source GeoPackage and exit",
    )

    args = parser.parse_args()

    if args.list_layers:
        if not args.gpkg.exists():
            print(f"Error: GeoPackage not found at {args.gpkg}")
            sys.exit(1)
        layers = list_gpkg_layers(args.gpkg)
        print(f"Layers in {args.gpkg.name}:")
        for lyr in layers:
            print(f" - {lyr}")
        sys.exit(0)

    custom_layers = args.layers.split(",") if args.layers else None
    custom_road_types = args.road_types.split(",") if args.road_types else None

    run_extraction(
        gpkg_path=args.gpkg,
        boundary_path=args.boundary,
        output_path=args.output,
        layers=custom_layers,
        road_types=custom_road_types,
        all_road_types=args.all_road_types,
        all_layers=args.all_layers,
        all_road_layers=args.all_road_layers,
        clip_mode=args.clip_mode,
        overwrite=not args.no_overwrite,
    )


if __name__ == "__main__":
    main()
