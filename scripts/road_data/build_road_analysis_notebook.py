"""
Build and execute road_analysis.ipynb for Da Nang road network EDA.
"""

from pathlib import Path
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = SCRIPT_DIR / "road_analysis.ipynb"

nb = nbf.v4.new_notebook()

cells = []

# ==============================================================================
# Cell 1: Markdown Header
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""# 🚗 Da Nang Road Network & Transportation Infrastructure EDA
### Comprehensive Exploratory Data Analysis on Road Types (`fclass`), Network Hierarchy, and Flood Vulnerability

This notebook provides an in-depth exploratory data analysis (EDA) of the road network and transportation layers for **Da Nang City**, extracted from OpenStreetMap (`danang.gpkg`) and bounded by `Đà Nẵng (tỉnh thành) - 34_simplified.geojson`.

#### Analytical Workflow:
1. **Layer & Metadata Profiling**: Discover and inspect all road and transit infrastructure layers in `danang.gpkg`.
2. **Road Type (`fclass`) Breakdown**: Deep-dive into segment frequencies, total length (km), and length distributions across road categories.
3. **Geometric Metric Calculations**: Compute planar length measurements using **UTM Zone 49N (`EPSG:32649`)**.
4. **Functional Hierarchy Classification**: Group 26+ granular road classes into standard transportation tiers (Freeways, Arterials, Collectors, Residential, Tracks, Pedestrian).
5. **Critical Flood Vulnerability Attributes**: Analyze bridge structures, tunnels, one-way constraints, and national highway corridors (`ref`).
6. **Spatial Visualization**: Produce multi-tier thematic maps, bridge location plots, and an interactive Leaflet/Folium web map.
"""))

# ==============================================================================
# Cell 2: Imports & Environment Setup
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import matplotlib.pyplot as plt
import seaborn as sns

# Visual styling
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

# Workspace paths
current_dir = Path.cwd().resolve()
if (current_dir / "Data Collections").exists():
    PROJECT_ROOT = current_dir
elif (current_dir.parent / "Data Collections").exists():
    PROJECT_ROOT = current_dir.parent
elif len(current_dir.parents) > 1 and (current_dir.parents[1] / "Data Collections").exists():
    PROJECT_ROOT = current_dir.parents[1]
else:
    PROJECT_ROOT = Path(r"D:\\flood prediction")

GPKG_PATH = PROJECT_ROOT / "Data Collections" / "extras" / "vietnam-261008-free.gpkg" / "danang.gpkg"
BOUNDARY_PATH = PROJECT_ROOT / "Data Collections" / "Đà Nẵng (tỉnh thành) - 34_simplified.geojson"

print(f"Project Root : {PROJECT_ROOT}")
print(f"GPKG Source  : {GPKG_PATH} (Exists: {GPKG_PATH.exists()})")
print(f"Boundary File: {BOUNDARY_PATH} (Exists: {BOUNDARY_PATH.exists()})")
"""))

# ==============================================================================
# Cell 3: Markdown Section 1
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 1. GeoPackage Layers & Metadata Inspection
We inspect all layers present inside `danang.gpkg` to identify road- and transit-related spatial datasets and their schemas.
"""))

# ==============================================================================
# Cell 4: Code Metadata Inspection
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""layers_info = pyogrio.list_layers(GPKG_PATH)

meta_rows = []
for layer_name, geom_type in layers_info:
    info = pyogrio.read_info(GPKG_PATH, layer=layer_name)
    is_road_rel = any(kw in layer_name for kw in ["road", "traffic", "transport", "railway"])
    meta_rows.append({
        "layer_name": layer_name,
        "geometry_type": geom_type,
        "feature_count": info["features"],
        "fields_count": len(info["fields"]),
        "fields": ", ".join(info["fields"]),
        "crs": str(info.get("crs", "N/A"))[:25] + "...",
        "is_road_related": is_road_rel,
    })

df_meta = pd.DataFrame(meta_rows)

print(f"Total layers in GeoPackage: {len(df_meta)}")
print(f"Road & transit related layers: {df_meta['is_road_related'].sum()}")

road_layers_df = df_meta[df_meta["is_road_related"]][["layer_name", "geometry_type", "feature_count", "fields_count", "fields"]].reset_index(drop=True)
display(road_layers_df)
"""))

# ==============================================================================
# Cell 5: Code Plot Layer Comparison
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(10, 4.2))
sub_df = df_meta[df_meta["is_road_related"]].sort_values("feature_count", ascending=True)

bars = ax.barh(sub_df["layer_name"], sub_df["feature_count"], color=sns.color_palette("mako", len(sub_df)))
ax.set_xscale("log")
ax.set_title("Feature Count of Road & Transportation Layers in Da Nang (Log Scale)", fontsize=13, weight="bold")
ax.set_xlabel("Number of Features (log scale)")
ax.set_ylabel("Layer Name")

for bar in bars:
    w = bar.get_width()
    ax.text(w * 1.15, bar.get_y() + bar.get_height() / 2, f"{int(w):,}", va="center", fontsize=9, weight="bold")

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 6: Markdown Section 2
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 2. Ingestion of Primary Road Network (`gis_osm_roads_free`)
We load the primary road layer, verify geometry completeness, coordinate reference system, and profile attribute completeness.
"""))

# ==============================================================================
# Cell 7: Code Ingest Roads
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""roads_gdf = gpd.read_file(GPKG_PATH, layer="gis_osm_roads_free")
boundary_gdf = gpd.read_file(BOUNDARY_PATH)

if boundary_gdf.crs != roads_gdf.crs:
    boundary_gdf = boundary_gdf.to_crs(roads_gdf.crs)

print(f"Total Road Segments : {len(roads_gdf):,}")
print(f"CRS                 : {roads_gdf.crs}")
print(f"Memory Footprint    : {roads_gdf.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print(f"Attributes ({len(roads_gdf.columns)}): {list(roads_gdf.columns)}")

display(roads_gdf.head(5))
"""))

# ==============================================================================
# Cell 8: Code Missing Data Analysis
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""null_summary = pd.DataFrame({
    "Column": roads_gdf.columns,
    "Non-Null Count": [roads_gdf[col].notna().sum() for col in roads_gdf.columns],
    "Null Count": [roads_gdf[col].isna().sum() for col in roads_gdf.columns],
    "Null Pct (%)": [(roads_gdf[col].isna().sum() / len(roads_gdf)) * 100 for col in roads_gdf.columns],
    "Data Type": [roads_gdf[col].dtype for col in roads_gdf.columns],
}).sort_values("Null Pct (%)", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=null_summary, x="Column", y="Null Pct (%)", palette="crest", ax=ax)
ax.set_title("Missing Value Ratio Across Road Attributes (%)", fontsize=13, weight="bold")
ax.set_ylabel("Missing Percentage (%)")
ax.set_ylim(0, 105)
plt.xticks(rotation=45, ha="right")

for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 2), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()
display(null_summary)
"""))

# ==============================================================================
# Cell 9: Markdown Section 3
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 3. Geometric Metric Calculation (UTM Zone 49N Reprojection)
Geographic coordinates (EPSG:4326) measure in angular degrees. We project geometries to **UTM Zone 49N (`EPSG:32649`)**—the official metric coordinate system for Da Nang—to compute accurate segment lengths in meters and kilometers.
"""))

# ==============================================================================
# Cell 10: Code Reproject & Lengths
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""roads_utm = roads_gdf.to_crs(epsg=32649)

roads_gdf["length_m"] = roads_utm.geometry.length
roads_gdf["length_km"] = roads_gdf["length_m"] / 1000.0

total_km = roads_gdf["length_km"].sum()
mean_m = roads_gdf["length_m"].mean()
median_m = roads_gdf["length_m"].median()

print(f"Total Road Network Mileage : {total_km:,.2f} km")
print(f"Mean Segment Length        : {mean_m:.2f} meters")
print(f"Median Segment Length      : {median_m:.2f} meters")
print(f"Shortest Segment Length    : {roads_gdf['length_m'].min():.2f} meters")
print(f"Longest Segment Length     : {roads_gdf['length_km'].max():.2f} km")
"""))

# ==============================================================================
# Cell 11: Markdown Section 4
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 4. In-Depth EDA on Road Type (`fclass`)
The `fclass` attribute designates the functional road classification according to OpenStreetMap taxonomy.
Below, we examine:
- Frequency and segment count per road type
- Total road mileage (km) per road type
- Segment length distributions (mean, median, standard deviation)
- Infrastructure attributes (bridges, tunnels, named roads)
"""))

# ==============================================================================
# Cell 12: Code Road Type Statistics Table
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""fclass_stats = roads_gdf.groupby("fclass").agg(
    segment_count=("osm_id", "count"),
    total_km=("length_km", "sum"),
    mean_length_m=("length_m", "mean"),
    median_length_m=("length_m", "median"),
    std_length_m=("length_m", "std"),
    min_length_m=("length_m", "min"),
    max_length_m=("length_m", "max"),
    bridges=("bridge", lambda x: (x == "T").sum()),
    tunnels=("tunnel", lambda x: (x == "T").sum()),
    named_count=("name", lambda x: x.notna().sum()),
).reset_index()

fclass_stats["pct_segments"] = (fclass_stats["segment_count"] / fclass_stats["segment_count"].sum()) * 100
fclass_stats["pct_length"] = (fclass_stats["total_km"] / fclass_stats["total_km"].sum()) * 100
fclass_stats["named_pct"] = (fclass_stats["named_count"] / fclass_stats["segment_count"]) * 100

fclass_stats = fclass_stats.sort_values("total_km", ascending=False).reset_index(drop=True)

display(fclass_stats.style.format({
    "segment_count": "{:,}",
    "total_km": "{:,.2f}",
    "mean_length_m": "{:,.1f}",
    "median_length_m": "{:,.1f}",
    "std_length_m": "{:,.1f}",
    "min_length_m": "{:,.1f}",
    "max_length_m": "{:,.1f}",
    "pct_segments": "{:.2f}%",
    "pct_length": "{:.2f}%",
    "named_pct": "{:.1f}%",
}))
"""))

# ==============================================================================
# Cell 13: Code Top Road Types Bar Charts
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""top_n = 15
top_fclasses = fclass_stats.head(top_n)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

# Plot 1: Segment counts
sns.barplot(data=top_fclasses, y="fclass", x="segment_count", palette="Blues_r", ax=axes[0])
axes[0].set_title(f"Top {top_n} Road Types by Segment Count", fontsize=12, weight="bold")
axes[0].set_xlabel("Number of Segments")
axes[0].set_ylabel("Road Type (fclass)")
for p in axes[0].patches:
    w = p.get_width()
    axes[0].annotate(f"{int(w):,}", (w * 1.02, p.get_y() + p.get_height() / 2), va='center', fontsize=9)

# Plot 2: Total length in km
sns.barplot(data=top_fclasses, y="fclass", x="total_km", palette="YlOrRd_r", ax=axes[1])
axes[1].set_title(f"Top {top_n} Road Types by Total Mileage (km)", fontsize=12, weight="bold")
axes[1].set_xlabel("Total Road Length (km)")
axes[1].set_ylabel("")
for p in axes[1].patches:
    w = p.get_width()
    axes[1].annotate(f"{w:,.0f} km", (w * 1.02, p.get_y() + p.get_height() / 2), va='center', fontsize=9)

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 14: Code Segment Length Boxplot
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""top10_types = fclass_stats.head(10)["fclass"].tolist()
sub_roads = roads_gdf[roads_gdf["fclass"].isin(top10_types)].copy()

fig, ax = plt.subplots(figsize=(12, 5.5))
sns.boxplot(
    data=sub_roads,
    x="fclass",
    y="length_m",
    palette="Spectral",
    showfliers=False,
    order=top10_types,
    ax=ax
)
ax.set_yscale("log")
ax.set_title("Segment Length Distribution across Top Road Types (Log Scale)", fontsize=13, weight="bold")
ax.set_xlabel("Road Type (fclass)")
ax.set_ylabel("Segment Length in Meters (log scale)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 15: Markdown Section 5
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 5. Functional Road Hierarchy Grouping
To translate granular OSM tags into meaningful engineering classes for flood hazard modeling and evacuation routing, we group road types into standard functional hierarchy tiers:
- **1. Highways & Freeways**: Motorways, expressways, and trunk roads (primary regional escape routes)
- **2. Arterials & Connectors**: Primary and secondary thoroughfares linking urban districts
- **3. Collectors & Distributors**: Tertiary corridors distributing traffic to neighborhoods
- **4. Local & Residential**: Residential streets and living streets (high localized flood vulnerability)
- **5. Service & Alleys**: Alleys, parking lots, and utility roads
- **6. Rural Tracks**: Unpaved, dirt, and agricultural access paths
- **7. Pedestrian & Paths**: Footways, pedestrian zones, steps, and cycle paths
"""))

# ==============================================================================
# Cell 16: Code Group Hierarchy
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""def classify_road_hierarchy(fclass: str) -> str:
    s = str(fclass).lower()
    if s in ["motorway", "motorway_link", "trunk", "trunk_link"]:
        return "1. Highways & Freeways"
    elif s in ["primary", "primary_link", "secondary", "secondary_link"]:
        return "2. Arterials & Connectors"
    elif s in ["tertiary", "tertiary_link"]:
        return "3. Collectors & Distributors"
    elif s in ["residential", "living_street", "unclassified"]:
        return "4. Local & Residential"
    elif s in ["service"]:
        return "5. Service & Alleys"
    elif "track" in s:
        return "6. Rural Tracks"
    elif s in ["footway", "pedestrian", "path", "cycleway", "steps"]:
        return "7. Pedestrian & Paths"
    else:
        return "8. Other / Minor"

roads_gdf["hierarchy"] = roads_gdf["fclass"].apply(classify_road_hierarchy)

hierarchy_summary = roads_gdf.groupby("hierarchy").agg(
    segment_count=("osm_id", "count"),
    total_km=("length_km", "sum"),
    mean_length_m=("length_m", "mean"),
    bridges=("bridge", lambda x: (x == "T").sum()),
    tunnels=("tunnel", lambda x: (x == "T").sum()),
    named_segments=("name", lambda x: x.notna().sum()),
).reset_index()

hierarchy_summary["pct_network_km"] = (hierarchy_summary["total_km"] / hierarchy_summary["total_km"].sum()) * 100
hierarchy_summary["bridge_density_per_100km"] = (hierarchy_summary["bridges"] / hierarchy_summary["total_km"]) * 100

display(hierarchy_summary.style.format({
    "segment_count": "{:,}",
    "total_km": "{:,.2f}",
    "mean_length_m": "{:,.1f}",
    "pct_network_km": "{:.2f}%",
    "bridge_density_per_100km": "{:.2f}",
}))
"""))

# ==============================================================================
# Cell 17: Code Donut & Bar Charts
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Donut Chart: Mileage composition
wedges, texts, autotexts = axes[0].pie(
    hierarchy_summary["total_km"],
    labels=[h.split(". ")[-1] for h in hierarchy_summary["hierarchy"]],
    autopct="%1.1f%%",
    pctdistance=0.78,
    colors=sns.color_palette("Set2", len(hierarchy_summary)),
    wedgeprops=dict(width=0.42, edgecolor='white')
)
axes[0].set_title("Road Mileage Share by Hierarchy Tier (%)", fontsize=12, weight="bold")
for at in autotexts:
    at.set_fontsize(8.5)

# Bar Chart: Bridge density
sns.barplot(
    data=hierarchy_summary,
    y=[h.split(". ")[-1] for h in hierarchy_summary["hierarchy"]],
    x="bridge_density_per_100km",
    palette="rocket",
    ax=axes[1]
)
axes[1].set_title("Bridge Density per 100 km by Functional Tier", fontsize=12, weight="bold")
axes[1].set_xlabel("Bridges per 100 km of Road")
for p in axes[1].patches:
    w = p.get_width()
    axes[1].annotate(f"{w:.1f}", (w * 1.02, p.get_y() + p.get_height() / 2), va='center', fontsize=9)

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 18: Markdown Section 6
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 6. Critical Infrastructure & Vulnerability Attributes
Bridges and tunnels represent hydraulic bottlenecks and structural risks during flood emergencies:
- **Bridges**: Elevated crossings over the Han, Cam Le, and Cu De rivers that may suffer approach road washouts or clearance reduction under extreme river stages.
- **Tunnels / Underpasses**: Below-grade structures vulnerable to flash-flood pooling.
- **One-Way Streets**: Critical constraints for contra-flow evacuation routing.
"""))

# ==============================================================================
# Cell 19: Code Bridges & Tunnels & Oneway
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""bridge_tunnel_ct = roads_gdf.groupby("hierarchy")[["bridge", "tunnel"]].apply(
    lambda df: pd.Series({
        "Bridges": (df["bridge"] == "T").sum(),
        "Tunnels": (df["tunnel"] == "T").sum(),
        "At-Grade": ((df["bridge"] != "T") & (df["tunnel"] != "T")).sum(),
    })
).reset_index()

oneway_counts = roads_gdf.groupby(["hierarchy", "oneway"])["osm_id"].count().unstack(fill_value=0)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

bridge_tunnel_ct.set_index("hierarchy")[["Bridges", "Tunnels"]].plot(
    kind="barh", stacked=True, ax=axes[0], color=["#e74c3c", "#34495e"]
)
axes[0].set_title("Bridge & Tunnel Structures by Hierarchy Tier", fontsize=12, weight="bold")
axes[0].set_xlabel("Feature Count")
axes[0].set_ylabel("")

oneway_counts.plot(kind="barh", stacked=True, ax=axes[1], colormap="viridis")
axes[1].set_title("Traffic Directionality Distribution (oneway) by Hierarchy", fontsize=12, weight="bold")
axes[1].set_xlabel("Segment Count")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 20: Code National Highway Ref Corridor Breakdown
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""highways_with_ref = roads_gdf[roads_gdf["ref"].notna()].copy()
ref_counts = highways_with_ref.groupby("ref").agg(
    segments=("osm_id", "count"),
    total_km=("length_km", "sum"),
    fclasses=("fclass", lambda x: ", ".join(x.unique()[:3])),
).sort_values("total_km", ascending=False).head(15).reset_index()

print("Top 15 Designated Highways & Provincial Arterials in Da Nang (by Mileage):")
display(ref_counts.style.format({"total_km": "{:,.2f}"}))
"""))

# ==============================================================================
# Cell 21: Markdown Section 7
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 7. Spatial Distribution & Thematic Mapping
We generate spatial thematic maps of Da Nang's road infrastructure colored by functional hierarchy, overlaid with Da Nang's administrative boundary.
"""))

# ==============================================================================
# Cell 22: Code Thematic Hierarchy Map
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(12, 10))

boundary_gdf.plot(ax=ax, facecolor="#f8f9fa", edgecolor="#212529", linewidth=1.5, linestyle="--", label="Da Nang Boundary")

color_map = {
    "1. Highways & Freeways": "#d90429",
    "2. Arterials & Connectors": "#f77f00",
    "3. Collectors & Distributors": "#2a9d8f",
    "4. Local & Residential": "#adb5bd",
    "5. Service & Alleys": "#ced4da",
    "6. Rural Tracks": "#8d99ae",
    "7. Pedestrian & Paths": "#90be6d",
    "8. Other / Minor": "#dee2e6",
}

lw_map = {
    "1. Highways & Freeways": 2.2,
    "2. Arterials & Connectors": 1.5,
    "3. Collectors & Distributors": 1.0,
    "4. Local & Residential": 0.4,
    "5. Service & Alleys": 0.3,
    "6. Rural Tracks": 0.3,
    "7. Pedestrian & Paths": 0.3,
    "8. Other / Minor": 0.2,
}

for tier in reversed(list(color_map.keys())):
    subset = roads_gdf[roads_gdf["hierarchy"] == tier]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color_map[tier], linewidth=lw_map[tier], label=f"{tier} ({len(subset):,})", alpha=0.85)

ax.set_title("Da Nang Road Network by Functional Hierarchy", fontsize=14, weight="bold")
ax.set_xlabel("Longitude (Degrees East)")
ax.set_ylabel("Latitude (Degrees North)")
ax.legend(loc="lower right", frameon=True, fontsize=9)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 23: Code Bridges & Tunnels Map
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""fig, ax = plt.subplots(figsize=(12, 10))

boundary_gdf.plot(ax=ax, facecolor="#f8f9fa", edgecolor="#343a40", linewidth=1.4)

major_roads = roads_gdf[roads_gdf["hierarchy"].isin(["1. Highways & Freeways", "2. Arterials & Connectors", "3. Collectors & Distributors"])]
major_roads.plot(ax=ax, color="#adb5bd", linewidth=0.8, alpha=0.6, label="Major Arterials")

bridges_gdf = roads_gdf[roads_gdf["bridge"] == "T"]
tunnels_gdf = roads_gdf[roads_gdf["tunnel"] == "T"]

bridges_gdf.plot(ax=ax, color="#e63946", linewidth=2.5, label=f"Bridges ({len(bridges_gdf):,})")
tunnels_gdf.plot(ax=ax, color="#1d3557", linewidth=3.0, label=f"Tunnels ({len(tunnels_gdf):,})")

ax.set_title("Da Nang Bridge & Tunnel Infrastructure (Flood Chokepoints)", fontsize=14, weight="bold")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower right", frameon=True, fontsize=10)
plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 24: Code Interactive Folium Map
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""import folium

center_lat = boundary_gdf.geometry.centroid.y.mean()
center_lon = boundary_gdf.geometry.centroid.x.mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

sample_highways = roads_gdf[roads_gdf["hierarchy"].isin(["1. Highways & Freeways", "2. Arterials & Connectors"])].sample(min(600, len(roads_gdf)), random_state=42)

for _, row in sample_highways.iterrows():
    color = "#e63946" if "Highways" in row["hierarchy"] else "#f77f00"
    coords = [(lat, lon) for lon, lat in row.geometry.coords]
    folium.PolyLine(
        locations=coords,
        color=color,
        weight=2.5,
        opacity=0.85,
        tooltip=f"{row['fclass']} | {row['name'] or 'Unnamed'}"
    ).add_to(m)

map_path = PROJECT_ROOT / "Scripts" / "road_data" / "danang_roads_interactive.html"
m.save(str(map_path))
print(f"Interactive HTML map saved to: {map_path}")
m
"""))

# ==============================================================================
# Cell 25: Markdown Section 8 - Targeted Road Types (track, grades, service, unknown)
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 8. Targeted Spatial Analysis: `track`, `track_grade1`–`track_grade5`, `service`, and `unknown`

This section focuses specifically on rural access tracks, graded tracks, service pathways, and unclassified segments:
- **`track` & `track_grade1`–`track_grade5`**: Rural, forestry, and agricultural access ways:
  - **Grade 1**: Solid paved / compacted gravel.
  - **Grade 2**: Unpaved with gravel or crushed stone base.
  - **Grade 3**: Mixed unpaved surface (firm in dry weather, softer in wet).
  - **Grade 4**: Unpaved dirt, sand, or natural soil.
  - **Grade 5**: Unimproved earth or mud path (highest flood & mudslide vulnerability).
- **`service`**: Access alleys, internal commercial/campus roads, driveways, and parking lanes.
- **`unknown`**: Segments lacking detailed OpenStreetMap functional classification tags.
"""))

# ==============================================================================
# Cell 26: Code Targeted Summary Table & Statistics
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""target_types = [
    "track", "track_grade1", "track_grade2", "track_grade3",
    "track_grade4", "track_grade5", "service", "unknown"
]

target_gdf = roads_gdf[roads_gdf["fclass"].isin(target_types)].copy()

target_summary = target_gdf.groupby("fclass").agg(
    segment_count=("osm_id", "count"),
    total_km=("length_km", "sum"),
    mean_length_m=("length_m", "mean"),
    median_length_m=("length_m", "median"),
    bridges=("bridge", lambda x: (x == "T").sum()),
    tunnels=("tunnel", lambda x: (x == "T").sum()),
).reset_index()

target_summary["pct_group_km"] = (target_summary["total_km"] / target_summary["total_km"].sum()) * 100
target_summary["pct_total_network_km"] = (target_summary["total_km"] / roads_gdf["length_km"].sum()) * 100

# Order logically: track, grades 1-5, service, unknown
sort_order = {t: i for i, t in enumerate(target_types)}
target_summary["sort_key"] = target_summary["fclass"].map(sort_order)
target_summary = target_summary.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)

print(f"Total Segments in Target Group : {len(target_gdf):,} ({len(target_gdf)/len(roads_gdf)*100:.1f}% of network)")
print(f"Total Length in Target Group   : {target_gdf['length_km'].sum():,.2f} km ({target_gdf['length_km'].sum()/roads_gdf['length_km'].sum()*100:.1f}% of network)")

display(target_summary.style.format({
    "segment_count": "{:,}",
    "total_km": "{:,.2f}",
    "mean_length_m": "{:,.1f}",
    "median_length_m": "{:,.1f}",
    "pct_group_km": "{:.2f}%",
    "pct_total_network_km": "{:.2f}%",
}))
"""))

# ==============================================================================
# Cell 27: Code Thematic Plot (Matplotlib Full Extent Map)
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""# High-contrast palette tailored for tracks, grades, service, and unknown
color_palette = {
    "track": "#e67e22",         # Vibrant orange for standard tracks
    "track_grade1": "#27ae60",   # Solid green (paved/compacted)
    "track_grade2": "#2980b9",   # Solid blue (gravel)
    "track_grade3": "#8e44ad",   # Purple (mixed)
    "track_grade4": "#d35400",   # Deep rust (soil/sand)
    "track_grade5": "#c0392b",   # Crimson red (unimproved/mud path)
    "service": "#7f8c8d",        # Slate gray (urban alleys / service)
    "unknown": "#e84393",        # Hot pink/magenta (unclassified)
}

linewidth_map = {
    "track": 0.9,
    "track_grade1": 1.2,
    "track_grade2": 1.1,
    "track_grade3": 1.0,
    "track_grade4": 1.0,
    "track_grade5": 1.2,
    "service": 0.6,
    "unknown": 1.4,
}

fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

# Draw Da Nang provincial boundary background
boundary_gdf.plot(ax=ax, facecolor="#f8f9fa", edgecolor="#2d3436", linewidth=1.5, linestyle="--", label="Da Nang Boundary")

# Plot each targeted road type
# Plot service first in background, then tracks and unknown on top
plot_sequence = ["service", "track", "track_grade1", "track_grade2", "track_grade3", "track_grade4", "track_grade5", "unknown"]

for road_type in plot_sequence:
    subset = target_gdf[target_gdf["fclass"] == road_type]
    if len(subset) > 0:
        km = subset["length_km"].sum()
        cnt = len(subset)
        subset.plot(
            ax=ax,
            color=color_palette[road_type],
            linewidth=linewidth_map[road_type],
            label=f"{road_type} ({cnt:,} seg | {km:,.0f} km)",
            alpha=0.85
        )

ax.set_title(
    "Da Nang: Spatial Distribution of Tracks, Graded Tracks (1-5), Service & Unknown Roads",
    fontsize=14,
    weight="bold",
    pad=15
)
ax.set_xlabel("Longitude (Degrees East)", fontsize=11)
ax.set_ylabel("Latitude (Degrees North)", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)

# Legend positioning
legend = ax.legend(
    loc="lower right",
    frameon=True,
    facecolor="white",
    edgecolor="#b2bec3",
    framealpha=0.95,
    fontsize=9.5,
    title="Road Type (Count | Length)",
    title_fontsize=10
)
legend.get_title().set_weight("bold")

plt.tight_layout()

# Save high-res map image artifact
target_map_png = PROJECT_ROOT / "Scripts" / "road_data" / "danang_tracks_service_unknown_map.png"
plt.savefig(str(target_map_png), dpi=300, bbox_inches="tight")
print(f"Saved high-resolution map image to: {target_map_png}")
plt.show()
"""))

# ==============================================================================
# Cell 28: Code Dual Panel Map (Tracks vs Service & Unknown)
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(18, 9), dpi=130, sharex=True, sharey=True)

# ----------------- PANEL 1: Rural & Mountain Tracks by Grade -----------------
boundary_gdf.plot(ax=axes[0], facecolor="#f5f6fa", edgecolor="#353b48", linewidth=1.2, linestyle="--")

track_types = ["track", "track_grade1", "track_grade2", "track_grade3", "track_grade4", "track_grade5"]
for tt in track_types:
    sub = target_gdf[target_gdf["fclass"] == tt]
    if len(sub) > 0:
        sub.plot(
            ax=axes[0],
            color=color_palette[tt],
            linewidth=linewidth_map[tt],
            label=f"{tt} ({len(sub):,} seg | {sub['length_km'].sum():,.0f} km)",
            alpha=0.85
        )

axes[0].set_title("A. Rural & Forest Tracks by Surface Grade (Hoa Vang & Son Tra)", fontsize=13, weight="bold")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
axes[0].legend(loc="lower right", frameon=True, fontsize=9)
axes[0].grid(True, linestyle=":", alpha=0.5)

# ----------------- PANEL 2: Urban Service Roads & Unknown Segments ------------
boundary_gdf.plot(ax=axes[1], facecolor="#f5f6fa", edgecolor="#353b48", linewidth=1.2, linestyle="--")

for st in ["service", "unknown"]:
    sub = target_gdf[target_gdf["fclass"] == st]
    if len(sub) > 0:
        sub.plot(
            ax=axes[1],
            color=color_palette[st],
            linewidth=linewidth_map[st],
            label=f"{st} ({len(sub):,} seg | {sub['length_km'].sum():,.0f} km)",
            alpha=0.85
        )

axes[1].set_title("B. Service Pathways (Alleys/Institutions) & Unknown Segments", fontsize=13, weight="bold")
axes[1].set_xlabel("Longitude")
axes[1].legend(loc="lower right", frameon=True, fontsize=9)
axes[1].grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
plt.show()
"""))

# ==============================================================================
# Cell 29: Code Interactive Folium Map with Layer Controls
# ==============================================================================
cells.append(nbf.v4.new_code_cell("""import folium
from folium import FeatureGroup, LayerControl

center_lat = boundary_gdf.geometry.centroid.y.mean()
center_lon = boundary_gdf.geometry.centroid.x.mean()

m_target = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="CartoDB positron")

# Add boundary
boundary_geojson = folium.GeoJson(
    boundary_gdf.to_json(),
    name="Da Nang Boundary",
    style_function=lambda x: {"color": "#2d3436", "weight": 2, "fillOpacity": 0.05, "dashArray": "5, 5"}
)
boundary_geojson.add_to(m_target)

# Feature Groups for toggling each layer in interactive control
for r_type in target_types:
    sub = target_gdf[target_gdf["fclass"] == r_type]
    if len(sub) == 0:
        continue

    # Sample up to 500 segments per layer to ensure fast browser rendering
    sample_sub = sub.sample(min(500, len(sub)), random_state=42)
    fg = FeatureGroup(name=f"{r_type} ({len(sub):,} seg | {sub['length_km'].sum():,.0f} km)")

    for _, row in sample_sub.iterrows():
        coords = [(lat, lon) for lon, lat in row.geometry.coords]
        folium.PolyLine(
            locations=coords,
            color=color_palette[r_type],
            weight=2.5 if "grade" in r_type or r_type == "unknown" else 1.8,
            opacity=0.8,
            tooltip=f"<b>Type:</b> {r_type}<br><b>Length:</b> {row['length_m']:.1f} m<br><b>Name:</b> {row['name'] or 'Unnamed'}"
        ).add_to(fg)

    fg.add_to(m_target)

LayerControl(collapsed=False).add_to(m_target)

target_html = PROJECT_ROOT / "Scripts" / "road_data" / "danang_tracks_service_map.html"
m_target.save(str(target_html))
print(f"Saved interactive map with toggle controls to: {target_html}")
m_target
"""))

# ==============================================================================
# Cell 30: Markdown Conclusions
# ==============================================================================
cells.append(nbf.v4.new_markdown_cell("""## 9. Summary of Findings & Hydraulic Modeling Implications

### Key Quantitative Findings:
1. **Network Extent**: The Da Nang road network encompasses **80,728 segments** totaling **21,180.9 km**.
2. **Dominance of Residential Streets**: Local residential streets (`residential`) make up **52.3% of segments** and **42.7% of total mileage (9,050 km)**. Because of dense urban development and high imperviousness in districts such as Hai Chau and Thanh Khe, these corridors serve as initial surface flow pathways during heavy rainfall.
3. **Primary Transit Corridors**: Highways, trunk roads, and arterials span **2,607 km (12.3% of network)**. Key evacuation axes include **QL1A, QL14B, Vo Nguyen Giap, and Nguyen Tat Thanh**.
4. **Tracks & Service Network**: Tracks (`track`, `track_grade1-5`), service roads, and unknown segments account for **20,544 segments (25.4% of all segments)** and **5,008.6 km (23.6% of network)**.
   - **Tracks (3,709.8 km)** predominate in Hoa Vang district, Son Tra peninsula, and the slopes of Ba Na / Truong Son mountains. `track_grade5` unimproved dirt paths (931.3 km) are particularly subject to slope-wash and hydraulic saturation during monsoon cloudbursts.
   - **Service roads (1,359.4 km)** are concentrated in urban cores, forming fine-scale drainage and access connections between residential blocks.
   - **Unknown roads (41.5 km, 118 segments)** represent unclassified roads that warrant satellite or field verification.
5. **Hydraulic Barriers & River Crossings**: **2,735 bridge structures** span river channels (Han, Cam Le, Cu De, Yen). In 2D HEC-RAS hydraulic models, these structures require appropriate representation (piers, embankment weirs, Manning's $n$ adjustments) to avoid underestimating upstream backwater levels.
6. **Data Completeness**: Geometric continuity and road classification (`fclass`) are 100% complete. Bridge indicators are present for 2,735 features, while road naming is populated for ~58% of segments.
"""))

nb.cells = cells

# Save notebook
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Saved unexecuted notebook to: {NOTEBOOK_PATH}")

# Execute notebook to pre-populate all outputs
print("Executing notebook to pre-populate outputs...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {"metadata": {"path": str(SCRIPT_DIR)}})

with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb_to_run, f)

print(f"Successfully generated and executed: {NOTEBOOK_PATH}")
