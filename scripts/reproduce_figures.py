import os
import json
import pandas as pd
from shapely.geometry import LineString
from pathlib import Path
import generate_charts

FIGURES_DIR = Path("figures")

print("1. Loading dataset...")
df = pd.read_excel("dataset/feature_table.xlsx")
geoms_wkt = df['geometry_wkt'].tolist()
flood_labels = (df['passability'] == 2).astype(int).tolist()

# 1. Regenerate Fig 1
print("2. Generating Fig 1...")
fig1_path = str(FIGURES_DIR / "fig1_study_area_floods.svg")
generate_charts.generate_fig1_study_area(geoms_wkt, flood_labels, fig1_path)

# 2. Regenerate Fig 4
print("3. Generating Fig 4...")
with open(FIGURES_DIR / "fig4_probs.json", "r", encoding="utf-8") as f:
    probs = json.load(f)
fig4_path = str(FIGURES_DIR / "fig4_risk_map.svg")
generate_charts.generate_fig4_risk_map(geoms_wkt, probs, fig4_path)

# 3. Regenerate Fig 8
print("4. Generating Fig 8...")
lambda_curves_data = {
    'Shortest time': [{'time_pct': 0.0, 'flood_dist': 1063.6}],
    'LR': [
        {'time_pct': 0.0, 'flood_dist': 791.7},
        {'time_pct': 0.0, 'flood_dist': 704.4},
        {'time_pct': 2.67, 'flood_dist': 596.9},
        {'time_pct': 5.15, 'flood_dist': 584.4},
        {'time_pct': 5.57, 'flood_dist': 582.5}
    ],
    'SVM (RBF)': [
        {'time_pct': 0.0, 'flood_dist': 725.6},
        {'time_pct': 0.0, 'flood_dist': 619.2},
        {'time_pct': 12.33, 'flood_dist': 533.9},
        {'time_pct': 21.38, 'flood_dist': 495.0},
        {'time_pct': 23.98, 'flood_dist': 468.3}
    ],
    'Random Forest': [
        {'time_pct': 0.0, 'flood_dist': 773.9},
        {'time_pct': 0.0, 'flood_dist': 642.2},
        {'time_pct': 5.57, 'flood_dist': 537.5},
        {'time_pct': 13.10, 'flood_dist': 492.5},
        {'time_pct': 13.74, 'flood_dist': 480.6}
    ],
    'XGBoost': [
        {'time_pct': 0.0, 'flood_dist': 746.4},
        {'time_pct': 0.0, 'flood_dist': 609.7},
        {'time_pct': 7.31, 'flood_dist': 512.8},
        {'time_pct': 14.87, 'flood_dist': 478.3},
        {'time_pct': 22.37, 'flood_dist': 463.9}
    ],
    'MLP': [
        {'time_pct': 0.0, 'flood_dist': 741.1},
        {'time_pct': 2.13, 'flood_dist': 598.3},
        {'time_pct': 12.40, 'flood_dist': 505.8},
        {'time_pct': 22.01, 'flood_dist': 496.9},
        {'time_pct': 31.79, 'flood_dist': 490.0}
    ],
    'GraphSAGE': [
        {'time_pct': 0.0, 'flood_dist': 668.9},
        {'time_pct': 4.04, 'flood_dist': 561.4},
        {'time_pct': 12.90, 'flood_dist': 552.5},
        {'time_pct': 28.68, 'flood_dist': 484.7},
        {'time_pct': 30.57, 'flood_dist': 484.2}
    ],
    'Oracle': [
        {'time_pct': 0.0, 'flood_dist': 680.3},
        {'time_pct': 0.10, 'flood_dist': 466.4},
        {'time_pct': 12.82, 'flood_dist': 405.3},
        {'time_pct': 33.54, 'flood_dist': 323.1},
        {'time_pct': 44.52, 'flood_dist': 323.9}
    ]
}
fig8_path = str(FIGURES_DIR / "fig8_lambda_tradeoff_curves.svg")
generate_charts.generate_fig8_lambda_tradeoff(lambda_curves_data, fig8_path)

# 4. Regenerate Fig 9
print("5. Generating Fig 9...")
with open(FIGURES_DIR / "fig9_routes.json", "r", encoding="utf-8") as f:
    routes_raw = json.load(f)
fig9_routes = {
    k: [LineString(coords) for coords in v]
    for k, v in routes_raw.items()
}
print(f"Loaded route segments: Shortest={len(fig9_routes['Shortest Path'])}, XGB={len(fig9_routes['XGBoost (Flood-Aware)'])}, Oracle={len(fig9_routes['Oracle (Ground Truth)'])}")
fig9_path = str(FIGURES_DIR / "fig9_representative_route.svg")
generate_charts.generate_fig9_representative_route(geoms_wkt, flood_labels, fig9_routes, fig9_path)

print("6. All SVGs generated successfully!")
