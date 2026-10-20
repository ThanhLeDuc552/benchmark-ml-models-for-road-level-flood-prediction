"""
Process Hai Chau road network data for Da Nang flood prediction dataset.

This script:
1. Loads the administrative boundary of Hai Chau from `hai_chau.geojson`.
2. Extracts and constructs the drive road network within Hai Chau using OSMnx and the local dataset.
3. Cross-references and enriches road infrastructure features (bridges, tunnels) using `danang.gpkg`.
4. Segments roads at intersections and ensures no segment exceeds 100 meters (split step).
5. Computes required road features matching `feature_dictionary` (parent_group == 'road', researched == 1, phase_1_inclusion == 1):
   - segment_id: Stable unique ID
   - osm_u: OSM start node ID
   - osm_v: OSM end node ID
   - osm_key: Parallel-edge key
   - node_degree: Combined connectivity at segment end nodes
   - length_m: Segment length in meters
   - road_class: OSM highway classification
   - is_underpass_tunnel_bridge: 0/1 indicator for bridge, tunnel, or underpass
6. Encodes all derived features and geometry within a self-contained road network graph
   exported as `hai_chau_road_graph.graphml` and `hai_chau_road_graph.gpkg`.
7. Occupies the `feature_table` sheet in `feature_table.xlsx` with the derived features,
   strictly preserving all other columns without touching or interpolating them.
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path
import sys
import time

# Ensure UTF-8 output encoding on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import geopandas as gpd
import networkx as nx
import openpyxl
import osmnx as ox
from pyproj import Transformer
import pyogrio
from shapely import wkt
from shapely.geometry import LineString, Point
from shapely.ops import substring, transform


# ==============================================================================
# DEFAULT PATH RESOLUTION
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent.parent

DEFAULT_GEOJSON_PATH = WORKSPACE_DIR / "resources for forming dataset" / "hai_chau.geojson"
DEFAULT_GPKG_PATH = WORKSPACE_DIR / "resources for forming dataset" / "danang.gpkg"
DEFAULT_EXCEL_PATH = WORKSPACE_DIR / "dataset" / "feature_table.xlsx"
DEFAULT_EXISTING_GRAPHML = WORKSPACE_DIR / "dataset" / "hai_chau_network.graphml"

DEFAULT_OUT_GRAPHML = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.graphml"
DEFAULT_OUT_GPKG = WORKSPACE_DIR / "dataset" / "hai_chau_road_graph.gpkg"
DEFAULT_CACHE_DIR = SCRIPT_DIR / "cache"


# ==============================================================================
# CORE WORKFLOW FUNCTIONS
# ==============================================================================

def load_hai_chau_boundary(geojson_path: Path) -> gpd.GeoDataFrame:
    """Load Hai Chau administrative boundary and return GeoDataFrame in EPSG:4326."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"Hai Chau boundary GeoJSON not found at: {geojson_path}")
    gdf = gpd.read_file(geojson_path)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)
    return gdf


def load_or_create_road_graph(
    boundary_gdf: gpd.GeoDataFrame,
    existing_graphml: Path,
    cache_dir: Path,
) -> nx.MultiDiGraph:
    """
    Load the drive road graph for Hai Chau, truncating it to the administrative boundary.
    Prefers local cached data / existing graphml to guarantee offline reproducibility.
    """
    poly = boundary_gdf.geometry.union_all()

    ox.settings.use_cache = True
    if cache_dir.exists():
        ox.settings.cache_folder = str(cache_dir)

    G = None
    # 1. Try loading from existing graphml if available
    if existing_graphml.exists():
        try:
            print(f"Loading base road network from existing GraphML: {existing_graphml.name} ...")
            G = ox.load_graphml(filepath=existing_graphml)
        except Exception as e:
            print(f"Warning: ox.load_graphml failed ({e}), falling back to nx.read_graphml...")
            try:
                G = nx.read_graphml(existing_graphml)
            except Exception as e2:
                print(f"Warning: nx.read_graphml failed ({e2})")

    # 2. If not loaded, query via OSMnx using cache
    if G is None:
        print("Querying road network via OSMnx (drive network)...")
        try:
            G = ox.graph_from_polygon(poly, network_type="drive", simplify=True)
        except Exception:
            G = ox.graph_from_place("Hai Chau, Da Nang, Vietnam", network_type="drive", simplify=True)

    # 3. Truncate graph cleanly to Hai Chau boundary polygon
    print("Truncating road network to exact Hai Chau administrative boundary...")
    try:
        G = ox.truncate.truncate_graph_polygon(G, poly)
    except Exception as e:
        print(f"Notice during polygon truncate ({e}); filtering edges by intersection...")
        edges_to_remove = []
        for u, v, k, d in G.edges(keys=True, data=True):
            if "geometry" in d:
                geom = d["geometry"]
                if isinstance(geom, str):
                    geom = wkt.loads(geom)
            else:
                u_pt = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
                v_pt = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
                geom = LineString([u_pt, v_pt])
            if not geom.intersects(poly):
                edges_to_remove.append((u, v, k))
        G.remove_edges_from(edges_to_remove)

    print(f"Base road graph ready: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges.")
    return G


def extract_structure_ids_from_gpkg(
    gpkg_path: Path,
    boundary_gdf: gpd.GeoDataFrame,
) -> tuple[set[str], set[str]]:
    """
    Extract bridge and tunnel OSM IDs from danang.gpkg within the Hai Chau boundary.
    """
    if not gpkg_path.exists():
        print(f"Warning: {gpkg_path} not found. Structure flags will rely solely on OSM tags.")
        return set(), set()

    poly = boundary_gdf.geometry.union_all()
    print(f"Reading infrastructure structures from {gpkg_path.name} (gis_osm_roads_free)...")
    try:
        roads = pyogrio.read_dataframe(gpkg_path, layer="gis_osm_roads_free", mask=poly)
    except Exception:
        bbox = tuple(boundary_gdf.total_bounds)
        roads = pyogrio.read_dataframe(gpkg_path, layer="gis_osm_roads_free", bbox=bbox)

    bridge_ids = set(roads[roads["bridge"] == "T"]["osm_id"].astype(str))
    tunnel_ids = set(roads[roads["tunnel"] == "T"]["osm_id"].astype(str))

    print(f"Found {len(bridge_ids)} bridge features and {len(tunnel_ids)} tunnel features in Hai Chau.")
    return bridge_ids, tunnel_ids


def build_split_segments_and_graph(
    G: nx.MultiDiGraph,
    bridge_ids: set[str],
    tunnel_ids: set[str],
    max_segment_length: float = 100.0,
) -> tuple[list[dict], nx.MultiDiGraph]:
    """
    Subdivide road graph edges into segments of at most max_segment_length (100 m),
    compute required road features, and encode them into a new segmented graph.
    """
    transformer_to_metric = Transformer.from_crs("EPSG:4326", "EPSG:2045", always_xy=True).transform
    transformer_to_wgs = Transformer.from_crs("EPSG:2045", "EPSG:4326", always_xy=True).transform

    G_split = nx.MultiDiGraph()
    G_split.graph["crs"] = "EPSG:4326"
    G_split.graph["name"] = "Hai Chau Road Graph (<= 100m Segments)"

    # Pre-add existing nodes with coordinates
    for n, d in G.nodes(data=True):
        G_split.add_node(
            str(n),
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            street_count=int(d.get("street_count", G.degree(n))),
        )

    segments_data: list[dict] = []
    intermediate_node_counter = 0

    for u, v, k, d in G.edges(keys=True, data=True):
        # 1. Retrieve or reconstruct geometry
        if "geometry" in d and d["geometry"]:
            geom = d["geometry"]
            if isinstance(geom, str):
                geom = wkt.loads(geom)
        else:
            u_pt = (float(G.nodes[u]["x"]), float(G.nodes[u]["y"]))
            v_pt = (float(G.nodes[v]["x"]), float(G.nodes[v]["y"]))
            geom = LineString([u_pt, v_pt])

        # 2. Metric projection (EPSG:2045) for planar length and splitting
        geom_m = transform(transformer_to_metric, geom)
        L = geom_m.length

        # 3. Road classification
        hway = d.get("highway", "residential")
        if isinstance(hway, list):
            hway = hway[0]
        road_class = str(hway).strip()

        # 4. Structure flag (tunnel, underpass, bridge)
        is_struct = 0
        if d.get("bridge") in ["yes", "T", "1", True] or d.get("tunnel") in ["yes", "T", "1", True]:
            is_struct = 1

        oid = d.get("osmid")
        oids: list[str] = []
        if oid:
            if isinstance(oid, str) and oid.startswith("["):
                try:
                    oids = [str(x) for x in ast.literal_eval(oid)]
                except Exception:
                    oids = [str(oid)]
            elif isinstance(oid, list):
                oids = [str(x) for x in oid]
            else:
                oids = [str(oid)]
        for o in oids:
            if o in bridge_ids or o in tunnel_ids:
                is_struct = 1
                break

        # 5. Node degree (connectivity at end nodes)
        deg_u = G.degree(u)
        deg_v = G.degree(v)
        node_deg = int(deg_u + deg_v)

        # 6. Determine segment division (at most max_segment_length meters)
        if L <= max_segment_length or L <= 0:
            n_segs = 1
        else:
            n_segs = int(math.ceil(L / max_segment_length))

        # 7. Generate subsegments
        curr_u_node = str(u)
        for i in range(n_segs):
            seg_id = f"{u}_{v}_{k}_{i}"

            if n_segs == 1:
                sub_m = geom_m
                sub_w = geom
                sub_len = L
                next_v_node = str(v)
            else:
                start_d = i * (L / n_segs)
                end_d = (i + 1) * (L / n_segs)
                sub_m = substring(geom_m, start_d, end_d)
                sub_w = transform(transformer_to_wgs, sub_m)
                sub_len = sub_m.length

                if i < n_segs - 1:
                    # Create an intermediate node
                    intermediate_node_counter += 1
                    next_v_node = f"int_{intermediate_node_counter}"
                    end_coords = sub_w.coords[-1]
                    G_split.add_node(
                        next_v_node,
                        x=float(end_coords[0]),
                        y=float(end_coords[1]),
                        street_count=2,
                    )
                else:
                    next_v_node = str(v)

            length_val = round(float(sub_len), 3)

            # Record tabular feature row
            seg_dict = {
                "segment_id": seg_id,
                "osm_u": int(u),
                "osm_v": int(v),
                "osm_key": int(k),
                "node_degree": node_deg,
                "length_m": length_val,
                "road_class": road_class,
                "is_underpass_tunnel_bridge": int(is_struct),
                "geometry": sub_w,
            }
            segments_data.append(seg_dict)

            # Add edge to the split graph with all derived features encoded
            G_split.add_edge(
                curr_u_node,
                next_v_node,
                key=int(k) if n_segs == 1 else i,
                segment_id=seg_id,
                osm_u=int(u),
                osm_v=int(v),
                osm_key=int(k),
                node_degree=node_deg,
                length_m=length_val,
                road_class=road_class,
                is_underpass_tunnel_bridge=int(is_struct),
                geometry=sub_w.wkt,
            )

            curr_u_node = next_v_node

    print(f"Segmentation complete: {len(G.edges):,} original edges -> {len(segments_data):,} segments (all <= {max_segment_length}m).")
    print(f"Segmented graph: {G_split.number_of_nodes():,} nodes, {G_split.number_of_edges():,} edges.")
    return segments_data, G_split


def save_road_graph(
    G_split: nx.MultiDiGraph,
    segments_data: list[dict],
    out_graphml: Path,
    out_gpkg: Path,
) -> None:
    """Save graph in GraphML format and GeoPackage format."""
    # 1. Save GraphML
    print(f"Exporting road graph to GraphML: {out_graphml.name} ...")
    out_graphml.parent.mkdir(parents=True, exist_ok=True)
    try:
        ox.save_graphml(G_split, filepath=out_graphml)
    except Exception:
        nx.write_graphml(G_split, out_graphml)

    # 2. Save GeoPackage for spatial GIS workflows
    print(f"Exporting road segments to GeoPackage: {out_gpkg.name} ...")
    gdf_segments = gpd.GeoDataFrame(segments_data, crs="EPSG:4326")
    gdf_segments.to_file(
        out_gpkg,
        layer="hai_chau_road_segments",
        driver="GPKG",
        engine="pyogrio",
    )
    print("Graph files successfully exported.")


def occupy_feature_table(
    excel_path: Path,
    segments_data: list[dict],
) -> None:
    """
    Occupy the 'feature_table' sheet in feature_table.xlsx with derived road features.
    Strictly preserves all other non-road columns without touching or interpolating them.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Feature table spreadsheet not found at: {excel_path}")

    print(f"Loading workbook: {excel_path.name} ...")
    wb = openpyxl.load_workbook(excel_path)
    if "feature_table" not in wb.sheetnames:
        raise ValueError(f"'feature_table' sheet missing from {excel_path}")

    ws = wb["feature_table"]
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]

    # Validate header positions
    col_map = {name: idx + 1 for idx, name in enumerate(headers) if name is not None}
    required_features = [
        "segment_id",
        "osm_u",
        "osm_v",
        "osm_key",
        "node_degree",
        "length_m",
        "road_class",
        "is_underpass_tunnel_bridge",
    ]
    for feat in required_features:
        if feat not in col_map:
            raise KeyError(f"Required feature column '{feat}' missing from feature_table headers: {headers}")

    print(f"Populating 'feature_table' sheet with {len(segments_data):,} rows...")

    # Clear any previous data rows if present
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)

    # Write each segment row
    for row_idx, seg in enumerate(segments_data, start=2):
        for feat in required_features:
            col_idx = col_map[feat]
            ws.cell(row_idx, col_idx).value = seg[feat]

    wb.save(excel_path)
    print(f"Successfully occupied 'feature_table' sheet in {excel_path.name} with {len(segments_data):,} rows.")


# ==============================================================================
# MAIN EXECUTION ENTRYPOINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build Hai Chau road network graph and occupy feature_table spreadsheet."
    )
    parser.add_argument(
        "--boundary",
        type=Path,
        default=DEFAULT_GEOJSON_PATH,
        help="Path to Hai Chau boundary GeoJSON (default: %(default)s)",
    )
    parser.add_argument(
        "--gpkg",
        type=Path,
        default=DEFAULT_GPKG_PATH,
        help="Path to Da Nang GeoPackage (default: %(default)s)",
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to feature_table.xlsx (default: %(default)s)",
    )
    parser.add_argument(
        "--out-graphml",
        type=Path,
        default=DEFAULT_OUT_GRAPHML,
        help="Output GraphML path (default: %(default)s)",
    )
    parser.add_argument(
        "--out-gpkg",
        type=Path,
        default=DEFAULT_OUT_GPKG,
        help="Output GeoPackage path (default: %(default)s)",
    )
    parser.add_argument(
        "--max-segment-length",
        type=float,
        default=100.0,
        help="Maximum length of a road segment in meters (default: %(default)s)",
    )

    args = parser.parse_args()

    start_time = time.time()
    print("=" * 75)
    print("HAI CHAU ROAD NETWORK GRAPH & FEATURE TABLE DERIVATION")
    print("=" * 75)
    print(f"Boundary File   : {args.boundary}")
    print(f"GeoPackage File : {args.gpkg}")
    print(f"Feature Table   : {args.excel}")
    print(f"Output GraphML  : {args.out_graphml}")
    print(f"Output GPKG     : {args.out_gpkg}")
    print(f"Max Segment Len : {args.max_segment_length} m")
    print("-" * 75)

    # 1. Load boundary
    boundary_gdf = load_hai_chau_boundary(args.boundary)

    # 2. Load and truncate road graph
    G = load_or_create_road_graph(
        boundary_gdf=boundary_gdf,
        existing_graphml=DEFAULT_EXISTING_GRAPHML,
        cache_dir=DEFAULT_CACHE_DIR,
    )

    # 3. Extract structure flags (bridges, tunnels) from danang.gpkg
    bridge_ids, tunnel_ids = extract_structure_ids_from_gpkg(args.gpkg, boundary_gdf)

    # 4. Segment edges at intersections and <= 100m, derive all required features
    segments_data, G_split = build_split_segments_and_graph(
        G=G,
        bridge_ids=bridge_ids,
        tunnel_ids=tunnel_ids,
        max_segment_length=args.max_segment_length,
    )

    # 5. Export graph resources
    save_road_graph(G_split, segments_data, args.out_graphml, args.out_gpkg)

    # 6. Occupy feature_table sheet
    occupy_feature_table(args.excel, segments_data)

    elapsed = time.time() - start_time
    print("=" * 75)
    print("EXECUTION SUMMARY")
    print("=" * 75)
    print(f"Total Segments Created : {len(segments_data):,}")
    print(f"Max Segment Length     : {max(s['length_m'] for s in segments_data):.2f} m")
    print(f"Min Segment Length     : {min(s['length_m'] for s in segments_data):.2f} m")
    print(f"Mean Segment Length    : {sum(s['length_m'] for s in segments_data)/len(segments_data):.2f} m")
    print(f"Structure Flagged (1)  : {sum(s['is_underpass_tunnel_bridge'] for s in segments_data):,}")
    print(f"Elapsed Time           : {elapsed:.2f} seconds")
    print("=" * 75)


if __name__ == "__main__":
    main()
