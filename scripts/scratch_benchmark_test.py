import os
import sys
import time
import math
import random
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.stats as stats
import shapely.wkt as wkt
import networkx as nx

import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

import xgboost as xgb
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

print("Starting benchmark verification script...")

from pathlib import Path

# 1. Load Data
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
excel_path = str(WORKSPACE_DIR / 'dataset' / 'feature_table.xlsx')
graphml_path = str(WORKSPACE_DIR / 'dataset' / 'hai_chau_road_graph.graphml')

df = pd.read_excel(excel_path, sheet_name=0)
G = nx.read_graphml(graphml_path)
print(f"Loaded DataFrame: {len(df)} rows, Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 2. Compute edge betweenness centrality
for u, v, k, d in G.edges(keys=True, data=True):
    d['weight'] = float(d.get('length_m', 1.0))
ebc = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
ebc_lookup = {}
for (u, v, k), val in ebc.items():
    seg_id = G[u][v][k].get('segment_id')
    if seg_id:
        ebc_lookup[seg_id] = val

df['betweenness'] = df['segment_id'].map(ebc_lookup).fillna(0.0)

# 3. Target Variable Definition
# passability: 0=passable (<10cm), 1=flooded-passable (10-30cm), 2=impassable (>30cm)
# Primary benchmark: y = 1 if impassable (passability == 2, >30cm water depth)
df['y_primary'] = (df['passability'] == 2).astype(int)
print(f"Class distribution (y_primary - impassable >=30cm): {df['y_primary'].value_counts().to_dict()}")

# 4. Spatial Block CV Partitioning (500m square grid)
geoms = df['geometry_wkt'].apply(wkt.loads)
cx = np.array([g.centroid.x for g in geoms])
cy = np.array([g.centroid.y for g in geoms])

block_size = 500.0
bx = ((cx - cx.min()) // block_size).astype(int)
by = ((cy - cy.min()) // block_size).astype(int)
df['block_id'] = [f"{x}_{y}" for x, y in zip(bx, by)]

unique_blocks = df['block_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_blocks)
block_to_fold = {b: i % 5 for i, b in enumerate(unique_blocks)}
df['spatial_fold'] = df['block_id'].map(block_to_fold)

# Random 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
df['random_fold'] = -1
for f_idx, (_, test_idx) in enumerate(kf.split(df)):
    df.loc[test_idx, 'random_fold'] = f_idx

print(f"Spatial fold sizes: {df['spatial_fold'].value_counts().to_dict()}")
print(f"Random fold sizes: {df['random_fold'].value_counts().to_dict()}")

# 5. Feature Engineering
raw_num_cols = [
    'length_m', 'node_degree', 'betweenness', 'is_underpass_tunnel_bridge',
    'elev_min', 'elev_mean', 'slope_mean', 'tpi', 'depression_depth',
    'dist_river', 'rain_total', 'rain_peak_intensity'
]
cat_cols = ['road_class']
df_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
feature_cols = raw_num_cols + list(df_encoded.columns)
X_all = pd.concat([df[raw_num_cols], df_encoded], axis=1).values
y_all = df['y_primary'].values

# 6. Line Graph Construction for GraphSAGE
seg2idx = {sid: i for i, sid in enumerate(df['segment_id'])}
node2segs = defaultdict(list)
for u, v, k, d in G.edges(keys=True, data=True):
    sid = d.get('segment_id')
    if sid in seg2idx:
        idx = seg2idx[sid]
        node2segs[u].append(idx)
        node2segs[v].append(idx)

line_edges = set()
for node, s_indices in node2segs.items():
    for i in range(len(s_indices)):
        for j in range(i + 1, len(s_indices)):
            u_seg, v_seg = s_indices[i], s_indices[j]
            line_edges.add((u_seg, v_seg))
            line_edges.add((v_seg, u_seg))

edge_index = torch.tensor(list(line_edges), dtype=torch.long).t().contiguous()
print(f"Constructed Line Graph: {len(df)} nodes, {edge_index.size(1)} directed edges")

print("All components verified successfully!")
