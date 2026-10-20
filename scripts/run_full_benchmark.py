"""
Full benchmark script mirroring the IEEE paper for Da Nang road-level flood prediction.
Executes all models, evaluation metrics, statistical tests, SHAP, and downstream routing.
"""
import os
import sys
import time
import math
import random
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import scipy.stats as stats
import shapely.wkt as wkt
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.metrics as metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

import xgboost as xgb
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from pathlib import Path
WORKSPACE_DIR = Path(__file__).resolve().parent.parent

print("--- Step 1: Loading Data ---", flush=True)
excel_path = str(WORKSPACE_DIR / 'dataset' / 'feature_table.xlsx')
graphml_path = str(WORKSPACE_DIR / 'dataset' / 'hai_chau_road_graph.graphml')

df = pd.read_excel(excel_path, sheet_name=0)
G = nx.read_graphml(graphml_path)
print(f"Loaded {len(df)} segments from Excel and {G.number_of_nodes()} nodes, {G.number_of_edges()} edges from GraphML.", flush=True)

print("--- Step 2: Feature Engineering & Preprocessing ---", flush=True)
# Edge weights as float
for u, v, k, d in G.edges(keys=True, data=True):
    d['weight'] = float(d.get('length_m', 1.0))

print("Computing edge betweenness centrality...", flush=True)
ebc = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
ebc_lookup = {}
for (u, v, k), val in ebc.items():
    seg_id = G[u][v][k].get('segment_id')
    if seg_id:
        ebc_lookup[seg_id] = val
df['betweenness'] = df['segment_id'].map(ebc_lookup).fillna(0.0)

# Target: 0 passable (<10cm), 1 flooded-passable (10-30cm), 2 non-passable (>30cm)
# Primary benchmark: impassable (passability == 2) is the severe flood class
df['y_primary'] = (df['passability'] == 2).astype(int)
print("Target distribution (y_primary):", df['y_primary'].value_counts().to_dict(), flush=True)

# Spatial Block Partitioning (500m square grid)
geoms = df['geometry_wkt'].apply(wkt.loads)
cx = np.array([g.centroid.x for g in geoms])
cy = np.array([g.centroid.y for g in geoms])
block_size = 500.0
bx = ((cx - cx.min()) // block_size).astype(int)
by = ((cy - cy.min()) // block_size).astype(int)
df['block_id'] = [f"{x}_{y}" for x, y in zip(bx, by)]

unique_blocks = np.array(list(set(df['block_id'])))
rng = np.random.RandomState(SEED)
rng.shuffle(unique_blocks)
block_to_fold = {b: i % 5 for i, b in enumerate(unique_blocks)}
df['spatial_fold'] = df['block_id'].map(block_to_fold)

# Random 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
df['random_fold'] = -1
for f_idx, (_, test_idx) in enumerate(kf.split(df)):
    df.loc[test_idx, 'random_fold'] = f_idx

# Feature Columns
raw_num_cols = [
    'length_m', 'node_degree', 'betweenness', 'is_underpass_tunnel_bridge',
    'elev_min', 'elev_mean', 'slope_mean', 'tpi', 'depression_depth',
    'dist_river', 'rain_total', 'rain_peak_intensity'
]
cat_cols = ['road_class']
df_encoded = pd.get_dummies(df[cat_cols], drop_first=True, dtype=float)
feature_names = raw_num_cols + list(df_encoded.columns)
X_raw = pd.concat([df[raw_num_cols], df_encoded], axis=1).values
y_raw = df['y_primary'].values

# Dual Line Graph for GraphSAGE
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

edge_index_full = torch.tensor(list(line_edges), dtype=torch.long).t().contiguous()
print(f"Dual graph: {len(df)} nodes, {edge_index_full.size(1)} directed edges.", flush=True)

# Neural Architectures
class TorchMLP(nn.Module):
    def __init__(self, in_features, hidden=(64, 32)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden[0]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden[1], 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

class RoadGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
        self.out = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        h = self.relu(self.conv1(x, edge_index))
        h = self.dropout(h)
        h = self.relu(self.conv2(h, edge_index))
        return self.out(h).squeeze(-1)

print("--- Step 3: Running Cross-Validation Benchmarks ---", flush=True)
models_list = ['LR', 'SVM (RBF)', 'Random Forest', 'XGBoost', 'MLP', 'GraphSAGE']

def evaluate_predictions(y_true, y_prob):
    auc_pr = metrics.average_precision_score(y_true, y_prob)
    auc_roc = metrics.roc_auc_score(y_true, y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    brier = metrics.brier_score_loss(y_true, y_prob)
    return {'AUC-PR': auc_pr, 'AUC-ROC': auc_roc, 'F1': f1, 'Recall': recall, 'Brier': brier}

def train_and_eval(fold_col):
    oof_probs = {m: np.zeros(len(df)) for m in models_list}
    fold_metrics = {m: [] for m in models_list}
    
    for f in range(5):
        train_idx = np.where(df[fold_col] != f)[0]
        test_idx = np.where(df[fold_col] == f)[0]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_raw[train_idx])
        X_test_scaled = scaler.transform(X_raw[test_idx])
        
        y_train = y_raw[train_idx]
        y_test = y_raw[test_idx]
        pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))
        
        # 1. LR (liblinear)
        lr = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000, random_state=SEED)
        lr.fit(X_train_scaled, y_train)
        p_lr = lr.predict_proba(X_test_scaled)[:, 1]
        oof_probs['LR'][test_idx] = p_lr
        fold_metrics['LR'].append(evaluate_predictions(y_test, p_lr))
        
        # 2. SVM (RBF)
        svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED)
        svm.fit(X_train_scaled, y_train)
        p_svm = svm.predict_proba(X_test_scaled)[:, 1]
        oof_probs['SVM (RBF)'][test_idx] = p_svm
        fold_metrics['SVM (RBF)'].append(evaluate_predictions(y_test, p_svm))
        
        # 3. Random Forest
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=SEED, n_jobs=1)
        rf.fit(X_raw[train_idx], y_train)
        p_rf = rf.predict_proba(X_raw[test_idx])[:, 1]
        oof_probs['Random Forest'][test_idx] = p_rf
        fold_metrics['Random Forest'].append(evaluate_predictions(y_test, p_rf))
        
        # 4. XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            scale_pos_weight=pos_weight, random_state=SEED, eval_metric='logloss', n_jobs=1
        )
        xgb_model.fit(X_raw[train_idx], y_train)
        p_xgb = xgb_model.predict_proba(X_raw[test_idx])[:, 1]
        oof_probs['XGBoost'][test_idx] = p_xgb
        fold_metrics['XGBoost'].append(evaluate_predictions(y_test, p_xgb))
        
        # 5. MLP (PyTorch)
        mlp = TorchMLP(X_train_scaled.shape[1])
        mlp_opt = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=1e-4)
        pos_weight_t = torch.tensor([pos_weight], dtype=torch.float)
        mlp_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        
        x_tr_t = torch.tensor(X_train_scaled, dtype=torch.float)
        y_tr_t = torch.tensor(y_train, dtype=torch.float)
        x_te_t = torch.tensor(X_test_scaled, dtype=torch.float)
        
        mlp.train()
        for _ in range(80):
            mlp_opt.zero_grad()
            l = mlp_crit(mlp(x_tr_t), y_tr_t)
            l.backward()
            mlp_opt.step()
        mlp.eval()
        with torch.no_grad():
            p_mlp = torch.sigmoid(mlp(x_te_t)).numpy()
        oof_probs['MLP'][test_idx] = p_mlp
        fold_metrics['MLP'].append(evaluate_predictions(y_test, p_mlp))
        
        # 6. GraphSAGE
        X_all_scaled = scaler.fit_transform(X_raw)
        x_tensor = torch.tensor(X_all_scaled, dtype=torch.float)
        y_tensor = torch.tensor(y_raw, dtype=torch.float)
        
        gnn = RoadGraphSAGE(in_channels=X_all_scaled.shape[1], hidden_channels=32)
        optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
        
        gnn.train()
        train_mask = torch.tensor(train_idx, dtype=torch.long)
        for epoch in range(100):
            optimizer.zero_grad()
            logits = gnn(x_tensor, edge_index_full)
            loss = criterion(logits[train_mask], y_tensor[train_mask])
            loss.backward()
            optimizer.step()
            
        gnn.eval()
        with torch.no_grad():
            preds = torch.sigmoid(gnn(x_tensor, edge_index_full)).numpy()
        p_gnn = preds[test_idx]
        oof_probs['GraphSAGE'][test_idx] = p_gnn
        fold_metrics['GraphSAGE'].append(evaluate_predictions(y_test, p_gnn))
        
    return oof_probs, fold_metrics

print("Running Spatial Block CV...", flush=True)
spatial_oof, spatial_metrics = train_and_eval('spatial_fold')
print("Spatial Block CV completed.", flush=True)

print("Running Random CV...", flush=True)
random_oof, random_metrics = train_and_eval('random_fold')
print("Random CV completed.", flush=True)

print("--- Step 4: Compiling Table I (Classification Benchmark) ---", flush=True)
table_1_rows = []
families = {
    'LR': 'Linear',
    'SVM (RBF)': 'Kernel',
    'Random Forest': 'Tree ensemble',
    'XGBoost': 'Tree ensemble',
    'MLP': 'Neural',
    'GraphSAGE': 'Graph'
}

for m in models_list:
    sp_f = spatial_metrics[m]
    rd_f = random_metrics[m]
    
    auc_pr_m = np.mean([f['AUC-PR'] for f in sp_f])
    auc_pr_s = np.std([f['AUC-PR'] for f in sp_f])
    auc_roc_m = np.mean([f['AUC-ROC'] for f in sp_f])
    auc_roc_s = np.std([f['AUC-ROC'] for f in sp_f])
    f1_m = np.mean([f['F1'] for f in sp_f])
    f1_s = np.std([f['F1'] for f in sp_f])
    rec_m = np.mean([f['Recall'] for f in sp_f])
    rec_s = np.std([f['Recall'] for f in sp_f])
    brier_m = np.mean([f['Brier'] for f in sp_f])
    brier_s = np.std([f['Brier'] for f in sp_f])
    
    rd_auc_m = np.mean([f['AUC-ROC'] for f in rd_f])
    rd_auc_s = np.std([f['AUC-ROC'] for f in rd_f])
    optimism_gap = rd_auc_m - auc_roc_m
    
    table_1_rows.append({
        'Model': m,
        'Family': families[m],
        'AUC-PR': f"{auc_pr_m:.3f} +/- {auc_pr_s:.3f}",
        'AUC-ROC': f"{auc_roc_m:.3f} +/- {auc_roc_s:.3f}",
        'F1': f"{f1_m:.3f} +/- {f1_s:.3f}",
        'Recall (flooded)': f"{rec_m:.3f} +/- {rec_s:.3f}",
        'Brier score': f"{brier_m:.3f} +/- {brier_s:.3f}",
        'Random CV AUC-ROC': f"{rd_auc_m:.3f} +/- {rd_auc_s:.3f}",
        'Optimism Gap': f"{optimism_gap:+.3f}"
    })

table_1_df = pd.DataFrame(table_1_rows)
print("\n" + "="*80, flush=True)
print("TABLE I: CLASSIFICATION BENCHMARK UNDER SPATIAL BLOCK CV VS RANDOM CV", flush=True)
print("="*80, flush=True)
print(table_1_df.to_string(index=False), flush=True)

# Statistical Significance Tests
print("\n--- Statistical Significance Tests across Folds (AUC-PR) ---", flush=True)
auc_pr_by_model = [[f['AUC-PR'] for f in spatial_metrics[m]] for m in models_list]
friedman_stat, friedman_p = stats.friedmanchisquare(*auc_pr_by_model)
print(f"Friedman Test: Chi2 = {friedman_stat:.3f}, p-value = {friedman_p:.4f}", flush=True)

best_m = max(models_list, key=lambda m: np.mean([f['AUC-PR'] for f in spatial_metrics[m]]))
print(f"Best model by AUC-PR: {best_m}", flush=True)
for m in models_list:
    if m != best_m:
        w_stat, w_p = stats.wilcoxon([f['AUC-PR'] for f in spatial_metrics[best_m]],
                                     [f['AUC-PR'] for f in spatial_metrics[m]])
        print(f"Wilcoxon {best_m} vs {m}: W = {w_stat:.1f}, p = {w_p:.4f}", flush=True)

print("\n--- Step 5: Explainability with TreeSHAP ---", flush=True)
xgb_full = xgb.XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.08,
    scale_pos_weight=(len(y_raw) - sum(y_raw)) / max(1, sum(y_raw)),
    random_state=SEED, eval_metric='logloss', n_jobs=1
)
xgb_full.fit(X_raw, y_raw)
dmat_full = xgb.DMatrix(X_raw, label=y_raw, feature_names=feature_names)
shap_contribs = xgb_full.get_booster().predict(dmat_full, pred_contribs=True)
shap_values = shap_contribs[:, :-1]
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
print("\nTop 7 Flood Drivers by Mean |SHAP|:", flush=True)
print(feat_imp.head(7).to_string(index=False), flush=True)

print("\n--- Step 6: Downstream Flood-Aware Routing Benchmark ---", flush=True)
speed_lookup = {
    'primary': 50.0 / 3.6,
    'primary_link': 40.0 / 3.6,
    'secondary': 40.0 / 3.6,
    'secondary_link': 30.0 / 3.6,
    'tertiary': 30.0 / 3.6,
    'tertiary_link': 25.0 / 3.6,
    'residential': 20.0 / 3.6
}

seg_prob_map = {m: dict(zip(df['segment_id'], spatial_oof[m])) for m in models_list}
y_true_map = dict(zip(df['segment_id'], y_raw))

for u, v, k, d in G.edges(keys=True, data=True):
    sid = d.get('segment_id')
    length = float(d.get('length_m', 50.0))
    rc = d.get('road_class', 'residential')
    speed = speed_lookup.get(rc, 20.0 / 3.6)
    t_free = length / speed
    d['t_free'] = t_free
    d['length'] = length
    d['flooded'] = y_true_map.get(sid, 0)
    for m in models_list:
        d[f'prob_{m}'] = seg_prob_map[m].get(sid, 0.0)

# Generate 150 reachable OD pairs
nodes = list(G.nodes())
random.seed(SEED)
od_pairs = []
attempts = 0
while len(od_pairs) < 150 and attempts < 2000:
    attempts += 1
    o, d = random.sample(nodes, 2)
    if nx.has_path(G, o, d):
        sp_len = nx.shortest_path_length(G, o, d, weight='length')
        if sp_len >= 300:
            od_pairs.append((o, d))

print(f"Generated {len(od_pairs)} valid OD pairs for routing benchmark.", flush=True)

LAMBDA = 5.0
strategies = ['Shortest time'] + models_list + ['Oracle (observed)']

def compute_route(G, orig, dest, cost_attr):
    path = nx.dijkstra_path(G, orig, dest, weight=cost_attr)
    total_t = 0.0
    flooded_count = 0
    flooded_len = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = min(G[u][v].values(), key=lambda x: x[cost_attr])
        total_t += edge_data['t_free']
        if edge_data['flooded'] == 1:
            flooded_count += 1
            flooded_len += edge_data['length']
    return total_t, flooded_count, flooded_len

for u, v, k, d in G.edges(keys=True, data=True):
    t_f = d['t_free']
    d['cost_shortest'] = t_f
    d['cost_oracle'] = t_f * (1.0 + LAMBDA * d['flooded'])
    for m in models_list:
        d[f'cost_{m}'] = t_f * (1.0 + LAMBDA * d[f'prob_{m}'])

routing_results = {s: {'flooded_segs': [], 'flooded_len': [], 'time': []} for s in strategies}
no_safe_route_count = 0

for o, d in od_pairs:
    t_ora, fl_cnt_ora, fl_len_ora = compute_route(G, o, d, 'cost_oracle')
    if fl_cnt_ora > 0:
        no_safe_route_count += 1
        
    t_short, fl_cnt_short, fl_len_short = compute_route(G, o, d, 'cost_shortest')
    routing_results['Shortest time']['time'].append(t_short)
    routing_results['Shortest time']['flooded_segs'].append(fl_cnt_short)
    routing_results['Shortest time']['flooded_len'].append(fl_len_short)
    
    routing_results['Oracle (observed)']['time'].append(t_ora)
    routing_results['Oracle (observed)']['flooded_segs'].append(fl_cnt_ora)
    routing_results['Oracle (observed)']['flooded_len'].append(fl_len_ora)
    
    for m in models_list:
        t_m, fl_cnt_m, fl_len_m = compute_route(G, o, d, f'cost_{m}')
        routing_results[m]['time'].append(t_m)
        routing_results[m]['flooded_segs'].append(fl_cnt_m)
        routing_results[m]['flooded_len'].append(fl_len_m)

no_safe_pct = (no_safe_route_count / len(od_pairs)) * 100.0

table_2_rows = []
t_short_arr = np.array(routing_results['Shortest time']['time'])

for s in strategies:
    f_segs_mean = np.mean(routing_results[s]['flooded_segs'])
    f_len_mean = np.mean(routing_results[s]['flooded_len'])
    t_arr = np.array(routing_results[s]['time'])
    t_inc = ((t_arr - t_short_arr) / t_short_arr) * 100.0
    t_inc_med = np.median(t_inc)
    
    table_2_rows.append({
        'Strategy': s,
        'Flooded seg. per trip': f"{f_segs_mean:.2f}",
        'Flooded length (m)': f"{f_len_mean:.1f}",
        'Time increase (median, %)': f"{t_inc_med:.1f}%" if s != 'Shortest time' else "0.0%",
        'No safe route (%)': f"{no_safe_pct:.1f}%"
    })

table_2_df = pd.DataFrame(table_2_rows)
print("\n" + "="*80, flush=True)
print(f"TABLE II: ROUTING RESULTS OVER {len(od_pairs)} OD PAIRS (lambda = {LAMBDA})", flush=True)
print("="*80, flush=True)
print(table_2_df.to_string(index=False), flush=True)

print("\n--- Step 7: Generating and Saving Figures ---", flush=True)
figures_dir = WORKSPACE_DIR / 'figures'
os.makedirs(figures_dir, exist_ok=True)
sns.set_theme(style='whitegrid', font='DejaVu Sans')

# 1. ROC and PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
palette = sns.color_palette("tab10", len(models_list))

for idx, m in enumerate(models_list):
    fpr, tpr, _ = metrics.roc_curve(y_raw, spatial_oof[m])
    roc_auc = metrics.roc_auc_score(y_raw, spatial_oof[m])
    axes[0].plot(fpr, tpr, label=f"{m} (AUC={roc_auc:.3f})", color=palette[idx], lw=2)
    
    prec, rec, _ = metrics.precision_recall_curve(y_raw, spatial_oof[m])
    pr_auc = metrics.average_precision_score(y_raw, spatial_oof[m])
    axes[1].plot(rec, prec, label=f"{m} (AUC-PR={pr_auc:.3f})", color=palette[idx], lw=2)

axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
axes[0].set_title('Spatial Block CV: ROC Curves', fontsize=13, weight='bold')
axes[0].set_xlabel('False Positive Rate', fontsize=11)
axes[0].set_ylabel('True Positive Rate', fontsize=11)
axes[0].legend(loc='lower right', frameon=True)

baseline_pr = sum(y_raw) / len(y_raw)
axes[1].plot([0, 1], [baseline_pr, baseline_pr], 'k--', lw=1, label=f'No Skill ({baseline_pr:.2f})')
axes[1].set_title('Spatial Block CV: Precision-Recall Curves', fontsize=13, weight='bold')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].legend(loc='lower left', frameon=True)
plt.tight_layout()
fig.savefig(str(figures_dir / 'fig1_roc_pr_curves.png'), dpi=300)
plt.close(fig)

# 2. Optimism Gap Bar Chart
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(models_list))
width = 0.35
sp_aucs = [np.mean([f['AUC-ROC'] for f in spatial_metrics[m]]) for m in models_list]
rd_aucs = [np.mean([f['AUC-ROC'] for f in random_metrics[m]]) for m in models_list]

ax.bar(x - width/2, sp_aucs, width, label='Spatial Block CV', color='#1f77b4', alpha=0.9)
ax.bar(x + width/2, rd_aucs, width, label='Random CV (Leaked)', color='#ff7f0e', alpha=0.9)
ax.set_ylabel('AUC-ROC', fontsize=12)
ax.set_title('Optimism Gap: Random Cross-Validation vs Spatial Block Cross-Validation', fontsize=13, weight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, fontsize=10, rotation=15)
ax.set_ylim(0.5, 1.0)
ax.legend(frameon=True)
plt.tight_layout()
fig.savefig(str(figures_dir / 'fig2_optimism_gap.png'), dpi=300)
plt.close(fig)

# 3. SHAP Summary Plot
fig, ax = plt.subplots(figsize=(9, 5.5))
top_feats = feat_imp.head(10).iloc[::-1]
ax.barh(top_feats['feature'], top_feats['mean_abs_shap'], color='#2ca02c', alpha=0.85)
ax.set_xlabel('Mean |SHAP Value| (Impact on Flood Prediction)', fontsize=11)
ax.set_title('Global Feature Importance via TreeSHAP (XGBoost)', fontsize=13, weight='bold')
plt.tight_layout()
fig.savefig(str(figures_dir / 'fig3_shap_importance.png'), dpi=300)
plt.close(fig)

# 4. Routing Trade-off Chart
fig, ax = plt.subplots(figsize=(9, 5))
f_lens = [np.mean(routing_results[s]['flooded_len']) for s in strategies]
t_meds = [np.median(((np.array(routing_results[s]['time']) - t_short_arr) / t_short_arr) * 100.0) for s in strategies]

scatter = ax.scatter(t_meds, f_lens, c=range(len(strategies)), cmap='viridis', s=160, zorder=5, edgecolors='black')
for i, s in enumerate(strategies):
    ax.annotate(s, (t_meds[i] + 0.3, f_lens[i] + 5), fontsize=10, weight='semibold')

ax.set_xlabel('Median Travel Time Increase (%)', fontsize=11)
ax.set_ylabel('Average Flooded Distance per Trip (m)', fontsize=11)
ax.set_title('Downstream Routing Trade-off: Flood Exposure vs Travel Time Overhead', fontsize=13, weight='bold')
plt.tight_layout()
fig.savefig(str(figures_dir / 'fig4_routing_tradeoff.png'), dpi=300)
plt.close(fig)

# 5. Risk Map of Best Model
df['risk_prob_best'] = spatial_oof[best_m]
df['risk_class'] = pd.cut(df['risk_prob_best'], bins=[-0.01, 0.33, 0.66, 1.01], labels=['Low Risk', 'Medium Risk', 'High Risk'])

fig, ax = plt.subplots(figsize=(8, 10))
risk_colors = {'Low Risk': '#2b83ba', 'Medium Risk': '#fdae61', 'High Risk': '#d7191c'}
for r_class, col in risk_colors.items():
    sub = df[df['risk_class'] == r_class]
    for geom in sub['geometry_wkt']:
        line = wkt.loads(geom)
        coords = np.array(line.coords)
        ax.plot(coords[:, 0], coords[:, 1], color=col, lw=1.6 if r_class=='High Risk' else 1.0, alpha=0.85)

custom_lines = [plt.Line2D([0], [0], color=col, lw=3) for col in risk_colors.values()]
ax.legend(custom_lines, risk_colors.keys(), loc='upper left', frameon=True, fontsize=11)
ax.set_title(f'Da Nang (Hai Chau): Out-of-Fold Flood Risk Map ({best_m})', fontsize=13, weight='bold')
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
fig.savefig(str(figures_dir / 'fig5_risk_map.png'), dpi=300)
plt.close(fig)

print("All figures successfully generated and saved!", flush=True)
