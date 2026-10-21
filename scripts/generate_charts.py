"""
Dedicated visualization generator using SVG for publication-grade IEEE conference figures.
Generates vector SVG for all 10 figures of the research paper:
- Fig 1: Study Area, Road Network & Observed Flooded Segments in Hai Chau (White Background)
- Fig 2: Spatial Block CV ROC & Precision-Recall Curves across 6 models (External Legend, No Overlap)
- Fig 3: Optimism Gap: Random CV overstatement vs Spatial Block CV
- Fig 4: Road Network Out-of-Fold Flood Risk Vector Map (White Background, 3 Risk Tiers)
- Fig 5: TreeSHAP Feature Importance horizontal bar chart (Expanded Margins)
- Fig 6: Model Feature Ranking Heatmap across 6 ML Families
- Fig 7: Downstream Routing Trade-off: Flood Exposure vs Travel Time Overhead (Clean Side Legend)
- Fig 8: Downstream Routing Trade-off Curves across Risk Aversion lambda (Clipped Axis, Dynamic Range)
- Fig 9: Representative OD Route Comparison: Shortest Path vs Best Model vs Oracle (White Background)
- Fig 10: Sensitivity Analysis: (a) Block Size, (b) Uncertain Labels
"""
import os
import math
import numpy as np
import pandas as pd
import shapely.wkt as wkt
from pathlib import Path

OUT_DIR = str(Path(__file__).resolve().parent.parent / 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Publication Design Tokens
FONT_STACK = "Arial, Helvetica, Segoe UI, sans-serif"
BG_WHITE = "#ffffff"
TEXT_TITLE = "#0f172a"
TEXT_SUBTITLE = "#334155"
TEXT_MUTED = "#64748b"
BORDER_AXIS = "#475569"
GRID_LINE = "#f1f5f9"

# Unified Color Palette across all figures
COLOR_MAP = {
    'LR': '#2563eb',             # Strong Blue
    'SVM (RBF)': '#7c3aed',      # Royal Purple
    'Random Forest': '#059669',  # Emerald Green
    'XGBoost': '#ea580c',        # Warm Orange
    'MLP': '#0891b2',            # Cyan / Deep Teal
    'GraphSAGE': '#d97706',      # Amber / Gold
    'Shortest time': '#dc2626',  # Crimson Red
    'Oracle (observed)': '#16a34a', # Forest Green
    'Oracle': '#16a34a'
}


def generate_fig1_study_area(geoms_wkt, flood_labels, out_path_svg):
    """Fig. 1: Study area road network and observed flooded segments in Hai Chau."""
    lines = [wkt.loads(g) for g in geoms_wkt]
    all_coords = [pt for l in lines for pt in l.coords]
    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w, h = 680, 880
    ox, oy, pw, ph = 40, 75, 600, 765
    scale = min(pw / (maxx - minx), ph / (maxy - miny))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')
    
    # Title & Subtitle
    svg.append(f'<text x="{w/2}" y="32" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Hai Chau District, Da Nang: Road Network &amp; Inundation Ground Truth</text>')
    svg.append(f'<text x="{w/2}" y="50" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_MUTED}">Typhoon Son Ca (14 Oct 2022) | 2,059 Segments: 876 Flooded (&gt;30 cm), 1,183 Passable/Uncertain</text>')

    # 1. Non-flooded segments first (Light neutral gray)
    for line, fl in zip(lines, flood_labels):
        if fl == 0:
            pts = []
            for x, y in line.coords:
                sx = ox + (x - minx) * scale
                sy = oy + ph - (y - miny) * scale
                pts.append(f"{sx:.1f},{sy:.1f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#cbd5e1" stroke-width="1.1" opacity="0.85"/>')

    # 2. Flooded segments on top (Vibrant crimson red)
    for line, fl in zip(lines, flood_labels):
        if fl == 1:
            pts = []
            for x, y in line.coords:
                sx = ox + (x - minx) * scale
                sy = oy + ph - (y - miny) * scale
                pts.append(f"{sx:.1f},{sy:.1f}")
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="#dc2626" stroke-width="2.3" stroke-linecap="round" opacity="0.95"/>')

    # Legend (Clean white card)
    svg.append(f'<rect x="{ox + 15}" y="{oy + 15}" width="250" height="72" fill="{BG_WHITE}" stroke="#cbd5e1" stroke-width="1.2" rx="4" opacity="0.96"/>')
    svg.append(f'<line x1="{ox + 30}" y1="{oy + 38}" x2="{ox + 60}" y2="{oy + 38}" stroke="#cbd5e1" stroke-width="2.5"/>')
    svg.append(f'<text x="{ox + 70}" y="{oy + 42}" font-family="{FONT_STACK}" font-size="10.5" fill="{TEXT_SUBTITLE}">Passable / Non-Flooded (y = 0)</text>')
    svg.append(f'<line x1="{ox + 30}" y1="{oy + 62}" x2="{ox + 60}" y2="{oy + 62}" stroke="#dc2626" stroke-width="3"/>')
    svg.append(f'<text x="{ox + 70}" y="{oy + 66}" font-family="{FONT_STACK}" font-size="10.5" font-weight="bold" fill="#dc2626">Observed Flooded (&gt;30 cm, y = 1)</text>')

    # North Indicator
    nx, ny = ox + pw - 30, oy + 35
    svg.append(f'<polygon points="{nx},{ny} {nx-5},{ny+14} {nx+5},{ny+14}" fill="{TEXT_TITLE}"/>')
    svg.append(f'<text x="{nx}" y="{ny-4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" font-weight="bold" fill="{TEXT_TITLE}">N</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")


def generate_fig2_roc_pr(roc_pr_data, out_path_svg):
    """Fig. 2: ROC and Precision-Recall curves under Spatial Block CV with clean external legend."""
    w, h = 1000, 510
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')

    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Classification Performance Under 5-Fold Spatial Block Cross-Validation</text>')

    panels = [
        {'ox': 70, 'oy': 60, 'pw': 390, 'ph': 330, 'title': '(a) Receiver Operating Characteristic (ROC)', 'xlabel': 'False Positive Rate', 'ylabel': 'True Positive Rate'},
        {'ox': 540, 'oy': 60, 'pw': 390, 'ph': 330, 'title': '(b) Precision-Recall Curves', 'xlabel': 'Recall (Flooded Class)', 'ylabel': 'Precision'}
    ]

    for p in panels:
        ox, oy, pw, ph = p['ox'], p['oy'], p['pw'], p['ph']
        svg.append(f'<text x="{ox + pw/2}" y="{oy - 14}" text-anchor="middle" font-family="{FONT_STACK}" font-size="12" font-weight="bold" fill="{TEXT_SUBTITLE}">{p["title"]}</text>')
        
        # Grid lines & ticks
        for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            tx = ox + tick * pw
            ty = oy + (1.0 - tick) * ph
            svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="{GRID_LINE}" stroke-width="1"/>')
            svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="{GRID_LINE}" stroke-width="1"/>')
            svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10.5" fill="{TEXT_MUTED}">{tick:.1f}</text>')
            svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10.5" fill="{TEXT_MUTED}">{tick:.1f}</text>')

        # Bounding box
        svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')
        svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 38}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">{p["xlabel"]}</text>')
        svg.append(f'<text x="{ox - 45}" y="{oy + ph/2}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}" transform="rotate(-90 {ox-45} {oy+ph/2})">{p["ylabel"]}</text>')

    # Reference Lines
    # 1. Diagonal on ROC
    svg.append(f'<line x1="70" y1="{60+330}" x2="{70+390}" y2="60" stroke="#94a3b8" stroke-dasharray="4,4" stroke-width="1.2"/>')
    # 2. Baseline on PR (0.425)
    baseline_y = 60 + (1.0 - 0.425) * 330
    svg.append(f'<line x1="540" y1="{baseline_y}" x2="{540+390}" y2="{baseline_y}" stroke="#94a3b8" stroke-dasharray="4,4" stroke-width="1.2"/>')
    svg.append(f'<text x="{540 + 390 - 8}" y="{baseline_y - 6}" text-anchor="end" font-family="{FONT_STACK}" font-size="9.5" fill="{TEXT_MUTED}">Baseline No-Skill (0.425)</text>')

    # Plot polylines
    for m_name, m_data in roc_pr_data.items():
        col = COLOR_MAP.get(m_name, '#333333')
        fpr, tpr = m_data['fpr'], m_data['tpr']
        pts_roc = [f"{70 + x*390:.1f},{60 + (1-y)*330:.1f}" for x, y in zip(fpr, tpr)]
        svg.append(f'<polyline points="{" ".join(pts_roc)}" fill="none" stroke="{col}" stroke-width="2.3" stroke-linecap="round"/>')

        rec, prec = m_data['rec'], m_data['prec']
        pts_pr = [f"{540 + x*390:.1f},{60 + (1-y)*330:.1f}" for x, y in zip(rec, prec)]
        svg.append(f'<polyline points="{" ".join(pts_pr)}" fill="none" stroke="{col}" stroke-width="2.3" stroke-linecap="round"/>')

    # External Unified Legend (Placed cleanly below panels, NO overlay with curves)
    svg.append(f'<rect x="70" y="435" width="860" height="58" fill="{BG_WHITE}" stroke="#cbd5e1" stroke-width="1" rx="4"/>')
    models_keys = list(roc_pr_data.keys())
    
    # Row 1 (first 3 models)
    for i in range(min(3, len(models_keys))):
        m = models_keys[i]
        col = COLOR_MAP.get(m, '#333333')
        lx = 95 + i * 280
        ly = 452
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        svg.append(f'<text x="{lx + 32}" y="{ly + 4}" font-family="{FONT_STACK}" font-size="10.5" font-weight="600" fill="{TEXT_SUBTITLE}">{m} <tspan font-weight="normal" fill="{TEXT_MUTED}">(ROC: {roc_pr_data[m]["auc_roc"]:.3f}, PR: {roc_pr_data[m]["auc_pr"]:.3f})</tspan></text>')

    # Row 2 (next 3 models)
    for i in range(3, min(6, len(models_keys))):
        m = models_keys[i]
        col = COLOR_MAP.get(m, '#333333')
        lx = 95 + (i - 3) * 280
        ly = 474
        svg.append(f'<line x1="{lx}" y1="{ly}" x2="{lx + 24}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        svg.append(f'<text x="{lx + 32}" y="{ly + 4}" font-family="{FONT_STACK}" font-size="10.5" font-weight="600" fill="{TEXT_SUBTITLE}">{m} <tspan font-weight="normal" fill="{TEXT_MUTED}">(ROC: {roc_pr_data[m]["auc_roc"]:.3f}, PR: {roc_pr_data[m]["auc_pr"]:.3f})</tspan></text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

# Backwards compatibility alias
generate_fig1_roc_pr = generate_fig2_roc_pr


def generate_fig3_optimism_gap(models, spatial_aucs, random_aucs, out_path_svg):
    """Fig. 3: Grouped bar chart for Spatial vs Random CV."""
    w, h = 820, 440
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')
    ox, oy, pw, ph = 75, 60, 680, 290

    svg.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Optimism Gap: Random CV Overstatement vs Spatial Block CV (AUC-ROC)</text>')

    # Grid lines & ticks
    for tick in np.linspace(0.7, 1.0, 7):
        ty = oy + (1.0 - (tick - 0.7) / 0.3) * ph
        svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10.5" fill="{TEXT_MUTED}">{tick:.2f}</text>')

    # Bounding box
    svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ox - 45}" y="{oy + ph/2}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}" transform="rotate(-90 {ox-45} {oy+ph/2})">AUC-ROC Score</text>')

    n = len(models)
    group_w = pw / n
    bar_w = group_w * 0.35

    for i, m in enumerate(models):
        gx = ox + i * group_w + group_w * 0.15
        sp_v = spatial_aucs[i]
        rd_v = random_aucs[i]
        gap = rd_v - sp_v

        h_sp = max(0, ((sp_v - 0.7) / 0.3) * ph)
        h_rd = max(0, ((rd_v - 0.7) / 0.3) * ph)

        # Spatial bar (Blue)
        bx_sp = gx
        by_sp = oy + ph - h_sp
        svg.append(f'<rect x="{bx_sp}" y="{by_sp}" width="{bar_w}" height="{h_sp}" fill="#2563eb" rx="2"/>')
        svg.append(f'<text x="{bx_sp + bar_w/2}" y="{by_sp - 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#1e40af">{sp_v:.3f}</text>')

        # Random bar (Orange)
        bx_rd = gx + bar_w + 4
        by_rd = oy + ph - h_rd
        svg.append(f'<rect x="{bx_rd}" y="{by_rd}" width="{bar_w}" height="{h_rd}" fill="#ea580c" rx="2"/>')
        svg.append(f'<text x="{bx_rd + bar_w/2}" y="{by_rd - 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#9a3412">{rd_v:.3f}</text>')

        # Optimism gap badge
        svg.append(f'<text x="{gx + bar_w + 2}" y="{min(by_sp, by_rd) - 16}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#dc2626">Δ+{gap:.3f}</text>')

        # Model label on x-axis
        svg.append(f'<text x="{gx + bar_w + 2}" y="{oy + ph + 20}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">{m}</text>')

    # Legend at bottom
    ly = oy + ph + 42
    svg.append(f'<rect x="{w/2 - 170}" y="{ly}" width="16" height="12" fill="#2563eb" rx="2"/>')
    svg.append(f'<text x="{w/2 - 146}" y="{ly + 10}" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_SUBTITLE}">Spatial Block CV (Unbiased Benchmark)</text>')
    svg.append(f'<rect x="{w/2 + 80}" y="{ly}" width="16" height="12" fill="#ea580c" rx="2"/>')
    svg.append(f'<text x="{w/2 + 104}" y="{ly + 10}" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_SUBTITLE}">Random CV (Optimistically Leaked)</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

# Backwards compatibility alias
generate_fig2_optimism_gap = generate_fig3_optimism_gap


def generate_fig4_risk_map(geoms_wkt, probs, out_path_svg):
    """Fig. 4: Vector road network risk map on clean white background."""
    lines = [wkt.loads(g) for g in geoms_wkt]
    all_coords = [pt for l in lines for pt in l.coords]
    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w, h = 680, 880
    ox, oy, pw, ph = 40, 75, 600, 765
    scale = min(pw / (maxx - minx), ph / (maxy - miny))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')
    
    # Title & Subtitle
    svg.append(f'<text x="{w/2}" y="32" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Da Nang (Hai Chau): Out-of-Fold Road Flood Risk Map</text>')
    svg.append(f'<text x="{w/2}" y="50" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_MUTED}">Predicted Inundation Probability Under 5-Fold Spatial Block Validation</text>')

    for line, p in zip(lines, probs):
        if p >= 0.66:
            col = '#dc2626' # High risk red
            lw = 2.4
            op = 0.95
        elif p >= 0.33:
            col = '#f59e0b' # Medium risk amber
            lw = 1.5
            op = 0.85
        else:
            col = '#2563eb' # Low risk royal blue
            lw = 1.0
            op = 0.75

        pts = []
        for x, y in line.coords:
            sx = ox + (x - minx) * scale
            sy = oy + ph - (y - miny) * scale
            pts.append(f"{sx:.1f},{sy:.1f}")
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="{lw}" stroke-linecap="round" opacity="{op}"/>')

    # Legend (Clean white card)
    svg.append(f'<rect x="{ox + 15}" y="{oy + 15}" width="220" height="90" fill="{BG_WHITE}" stroke="#cbd5e1" stroke-width="1.2" rx="4" opacity="0.96"/>')
    legend_tiers = [
        ('Low Risk (p &lt; 0.33)', '#2563eb', '1.6'),
        ('Medium Risk (0.33 ≤ p &lt; 0.66)', '#f59e0b', '2.2'),
        ('High Risk (p ≥ 0.66)', '#dc2626', '3.0')
    ]
    for idx, (label, col, lw_leg) in enumerate(legend_tiers):
        ly = oy + 36 + idx * 22
        svg.append(f'<line x1="{ox + 28}" y1="{ly}" x2="{ox + 55}" y2="{ly}" stroke="{col}" stroke-width="{lw_leg}"/>')
        svg.append(f'<text x="{ox + 65}" y="{ly + 4}" font-family="{FONT_STACK}" font-size="10.5" fill="{TEXT_SUBTITLE}">{label}</text>')

    # North Arrow
    nx, ny = ox + pw - 30, oy + 35
    svg.append(f'<polygon points="{nx},{ny} {nx-5},{ny+14} {nx+5},{ny+14}" fill="{TEXT_TITLE}"/>')
    svg.append(f'<text x="{nx}" y="{ny-4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" font-weight="bold" fill="{TEXT_TITLE}">N</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

# Backwards compatibility alias
generate_fig5_map = generate_fig4_risk_map


def generate_fig5_shap(feat_names, shap_vals, out_path_svg):
    """Fig. 5: Horizontal bar chart of SHAP values with expanded margins to prevent label clipping."""
    w, h = 740, 420
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')
    
    # ox = 190 ensures long feature names never clip
    ox, oy, pw, ph = 190, 55, 490, 310

    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Global Road Flood Predictor Importance via TreeSHAP (XGBoost)</text>')

    max_val = max(shap_vals) * 1.15
    for tick in np.linspace(0, max_val, 6):
        tx = ox + (tick / max_val) * pw
        svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{tick:.2f}</text>')

    svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 38}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">Mean Absolute SHAP Value: E[|SHAP|]</text>')

    n = len(feat_names)
    bar_h = ph / n - 8

    for i, (fn, val) in enumerate(zip(feat_names, shap_vals)):
        bw = (val / max_val) * pw
        by = oy + i * (ph / n) + 4
        svg.append(f'<rect x="{ox}" y="{by}" width="{bw}" height="{bar_h}" fill="#059669" rx="3"/>')
        # Feature label on the left (fully contained inside [0, ox])
        svg.append(f'<text x="{ox - 10}" y="{by + bar_h/2 + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10.5" font-weight="600" fill="{TEXT_SUBTITLE}">{fn}</text>')
        # Value label to the right of the bar
        svg.append(f'<text x="{ox + bw + 8}" y="{by + bar_h/2 + 4}" font-family="{FONT_STACK}" font-size="10" font-weight="bold" fill="#065f46">{val:.3f}</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

# Backwards compatibility alias
generate_fig3_shap = generate_fig5_shap


def generate_fig6_feature_ranks(models, feat_names, rank_df, out_path_svg):
    """Fig. 6: Heatmap of feature rank across all 6 models."""
    w, h = 820, 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')

    ox, oy = 190, 75
    n_feats = len(feat_names)
    cell_w = 95
    cell_h = 32

    svg.append(f'<text x="{w/2}" y="30" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Feature Importance Rank Consistency Across Six Machine Learning Families</text>')
    svg.append(f'<text x="{w/2}" y="48" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_MUTED}">Rank 1 = Top Predictor | Color intensity denotes relative importance tier</text>')

    # Column headers (Models)
    for j, m in enumerate(models):
        mx = ox + j * cell_w + cell_w / 2
        svg.append(f'<text x="{mx}" y="{oy - 12}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="bold" fill="{TEXT_SUBTITLE}">{m}</text>')

    # Rows (Features)
    for i, fn in enumerate(feat_names):
        ry = oy + i * cell_h
        svg.append(f'<text x="{ox - 12}" y="{ry + cell_h/2 + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">{fn}</text>')

        for j, m in enumerate(models):
            cx = ox + j * cell_w
            raw_rank = rank_df.loc[fn, m] if fn in rank_df.index and m in rank_df.columns else n_feats
            rank = int(raw_rank) if pd.notna(raw_rank) else n_feats
            
            # Tiered publication colors
            if rank == 1:
                bg_col = "#1d4ed8"
                text_col = "#ffffff"
            elif rank <= 3:
                bg_col = "#3b82f6"
                text_col = "#ffffff"
            elif rank <= 5:
                bg_col = "#93c5fd"
                text_col = "#1e293b"
            else:
                bg_col = "#e2e8f0"
                text_col = "#475569"

            svg.append(f'<rect x="{cx + 2}" y="{ry + 2}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{bg_col}" rx="4"/>')
            svg.append(f'<text x="{cx + cell_w/2}" y="{ry + cell_h/2 + 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10.5" font-weight="bold" fill="{text_col}">#{rank}</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")


def generate_fig7_routing(strategies, times, floods, out_path_svg):
    """Fig. 7: Scatter plot of routing trade-offs at lambda=5.0 with external side legend."""
    w, h = 880, 480
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')
    
    ox, oy, pw, ph = 80, 65, 520, 350

    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Downstream Routing Benchmark: Hazard Avoidance vs Travel-Time Overhead (λ = 5.0)</text>')
    svg.append(f'<text x="{w/2}" y="46" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_MUTED}">Evaluated Across 150 Representative Origin-Destination Trips in Hai Chau District</text>')

    max_t = 26.0
    min_fl, max_fl = 400.0, 1150.0

    for t_tick in [0, 5, 10, 15, 20, 25]:
        tx = ox + (t_tick / max_t) * pw
        svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{t_tick}%</text>')

    for fl_tick in [400, 600, 800, 1000]:
        ty = oy + ph - ((fl_tick - min_fl) / (max_fl - min_fl)) * ph
        svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{fl_tick}m</text>')

    svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 38}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">Median Travel Time Increase Over Shortest Path (%)</text>')
    svg.append(f'<text x="{ox - 50}" y="{oy + ph/2}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}" transform="rotate(-90 {ox-50} {oy+ph/2})">Mean Inundated Distance per Trip (m)</text>')

    # Dedicated Side Legend Box (x: 630..855) to eliminate label overlaps
    leg_x = 630
    svg.append(f'<rect x="{leg_x}" y="{oy}" width="225" height="{ph}" fill="{BG_WHITE}" stroke="#cbd5e1" stroke-width="1.2" rx="4"/>')
    svg.append(f'<text x="{leg_x + 15}" y="{oy + 24}" font-family="{FONT_STACK}" font-size="11" font-weight="bold" fill="{TEXT_TITLE}">Routing Strategy Results</text>')
    svg.append(f'<text x="{leg_x + 15}" y="{oy + 40}" font-family="{FONT_STACK}" font-size="9.5" fill="{TEXT_MUTED}">(Flooded length, Time overhead)</text>')

    for i, s in enumerate(strategies):
        t_val = times[i]
        fl_val = floods[i]
        cx = ox + (t_val / max_t) * pw
        cy = oy + ph - ((fl_val - min_fl) / (max_fl - min_fl)) * ph
        col = COLOR_MAP.get(s, '#333333')

        # Scatter dot on plot
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{col}" stroke="#ffffff" stroke-width="1.5"/>')
        
        # Compact scatter tag
        tag_text = "Shortest" if "Shortest" in s else ("Oracle" if "Oracle" in s else s)
        tag_y_offset = -12 if i % 2 == 0 else 16
        svg.append(f'<text x="{cx}" y="{cy + tag_y_offset}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9" font-weight="bold" fill="{col}">{tag_text}</text>')

        # Legend entry row
        ry = oy + 65 + i * 34
        svg.append(f'<circle cx="{leg_x + 22}" cy="{ry}" r="6" fill="{col}"/>')
        svg.append(f'<text x="{leg_x + 36}" y="{ry - 1}" font-family="{FONT_STACK}" font-size="10.5" font-weight="bold" fill="{TEXT_SUBTITLE}">{s}</text>')
        svg.append(f'<text x="{leg_x + 36}" y="{ry + 13}" font-family="{FONT_STACK}" font-size="9.5" fill="{TEXT_MUTED}">{fl_val:.1f} m  |  +{t_val:.1f}% time</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

# Backwards compatibility alias
generate_fig4_routing = generate_fig7_routing


def generate_fig8_lambda_tradeoff(lambda_curves_data, out_path_svg):
    """Fig. 8: Downstream routing trade-off curves with strict axis clipping and dynamic range."""
    w, h = 880, 490
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    
    # Clip path definition to prevent ANY leakage outside axis box
    ox, oy, pw, ph = 80, 65, 540, 360
    svg.append(f'''<defs>
  <clipPath id="fig8-plot-clip">
    <rect x="{ox}" y="{oy}" width="{pw}" height="{ph}"/>
  </clipPath>
</defs>''')
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')

    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Trade-off Curves: Flood Exposure vs Travel-Time Increase Across Risk Aversion λ ∈ [1, 20]</text>')
    svg.append(f'<text x="{w/2}" y="46" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_MUTED}">Parametric sweep: λ controls penalty severity C_i = t_i (1 + λ p_i)</text>')

    # Dynamically determine bounds to prevent out-of-bound coordinates
    all_times = [pt['time_pct'] for pts in lambda_curves_data.values() for pt in pts]
    all_floods = [pt['flood_dist'] for pts in lambda_curves_data.values() for pt in pts]
    
    max_t = max(35.0, math.ceil(max(all_times) * 1.15 / 5.0) * 5.0) if all_times else 40.0
    min_fl = max(0.0, math.floor(min(all_floods) * 0.85 / 100.0) * 100.0) if all_floods else 300.0
    max_fl = math.ceil(max(all_floods) * 1.10 / 100.0) * 100.0 if all_floods else 1200.0

    # X-axis ticks (Travel time %)
    t_ticks = np.linspace(0, max_t, int(max_t / 5) + 1)
    for t_tick in t_ticks:
        tx = ox + (t_tick / max_t) * pw
        svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{int(t_tick)}%</text>')

    # Y-axis ticks (Flooded distance m)
    fl_ticks = np.linspace(min_fl, max_fl, 6)
    for fl_tick in fl_ticks:
        ty = oy + ph - ((fl_tick - min_fl) / (max_fl - min_fl)) * ph
        svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{int(fl_tick)}m</text>')

    # Bounding box
    svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 38}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">Median Travel Time Overhead (%)</text>')
    svg.append(f'<text x="{ox - 50}" y="{oy + ph/2}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}" transform="rotate(-90 {ox-50} {oy+ph/2})">Flooded Distance per Trip (m)</text>')

    # Side Legend Box (x: 650..855) outside the plot bounds
    leg_x = 650
    svg.append(f'<rect x="{leg_x}" y="{oy}" width="205" height="{ph}" fill="{BG_WHITE}" stroke="#cbd5e1" stroke-width="1.2" rx="4"/>')
    svg.append(f'<text x="{leg_x + 15}" y="{oy + 24}" font-family="{FONT_STACK}" font-size="11" font-weight="bold" fill="{TEXT_TITLE}">Models &amp; Baselines</text>')
    svg.append(f'<text x="{leg_x + 15}" y="{oy + 40}" font-family="{FONT_STACK}" font-size="9.5" fill="{TEXT_MUTED}">Curves over λ ∈ [1, 20]</text>')

    # Render curves with STRICT CLIPPING
    svg.append('<g clip-path="url(#fig8-plot-clip)">')
    leg_idx = 0
    for s_name, points in lambda_curves_data.items():
        col = COLOR_MAP.get(s_name, '#333333')
        pts_svg = []
        for pt in points:
            t_pct, f_dist = pt['time_pct'], pt['flood_dist']
            cx = ox + (t_pct / max_t) * pw
            cy = oy + ph - ((f_dist - min_fl) / (max_fl - min_fl)) * ph
            pts_svg.append((cx, cy))
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{col}"/>')

        if len(pts_svg) > 1:
            line_str = " ".join([f"{x:.1f},{y:.1f}" for x, y in pts_svg])
            svg.append(f'<polyline points="{line_str}" fill="none" stroke="{col}" stroke-width="2.3" stroke-linecap="round"/>')
    svg.append('</g>')

    # Render Legend entries outside clipping
    for s_name in lambda_curves_data.keys():
        col = COLOR_MAP.get(s_name, '#333333')
        ly = oy + 66 + leg_idx * 34
        svg.append(f'<line x1="{leg_x + 15}" y1="{ly}" x2="{leg_x + 40}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        svg.append(f'<circle cx="{leg_x + 27.5}" cy="{ly}" r="3.5" fill="{col}"/>')
        svg.append(f'<text x="{leg_x + 48}" y="{ly + 4}" font-family="{FONT_STACK}" font-size="10.5" font-weight="600" fill="{TEXT_SUBTITLE}">{s_name}</text>')
        leg_idx += 1

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")


def generate_fig9_representative_route(geoms_wkt, flood_labels, routes_dict, out_path_svg):
    """Fig. 9: Representative OD pair comparison map on white background."""
    lines = [wkt.loads(g) for g in geoms_wkt]
    all_coords = [pt for l in lines for pt in l.coords]
    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]

    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)

    w, h = 760, 880
    ox, oy, pw, ph = 40, 75, 680, 765
    scale = min(pw / (maxx - minx), ph / (maxy - miny))

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')
    
    # Title & Subtitle
    svg.append(f'<text x="{w/2}" y="32" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Representative Origin-Destination Navigation: Shortest Path vs ML vs Oracle</text>')
    svg.append(f'<text x="{w/2}" y="50" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_MUTED}">Shortest path penetrates 14 flooded segments (951 m); flood-aware routing detours safely around inundation</text>')

    # 1. Base network on white background
    for line, fl in zip(lines, flood_labels):
        col = '#fecaca' if fl == 1 else '#cbd5e1'
        lw = 1.6 if fl == 1 else 0.9
        pts = [f"{ox + (x - minx) * scale:.1f},{oy + ph - (y - miny) * scale:.1f}" for x, y in line.coords]
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="{lw}" opacity="0.8"/>')

    # 2. Specific routes
    route_colors = {
        'Shortest Path': '#dc2626',           # Red: penetrates flood
        'XGBoost (Flood-Aware)': '#2563eb',   # Blue: safe detour
        'MLP (Flood-Aware)': '#0891b2',       # Cyan: safe detour
        'Oracle (Ground Truth)': '#16a34a'    # Green: optimal hazard avoidance
    }

    start_pt = None
    end_pt = None

    for r_label, seg_lines in routes_dict.items():
        col = route_colors.get(r_label, '#2563eb')
        for idx_s, s_geom in enumerate(seg_lines):
            pts = [f"{ox + (x - minx) * scale:.1f},{oy + ph - (y - miny) * scale:.1f}" for x, y in s_geom.coords]
            svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="3.5" stroke-linecap="round" opacity="0.95"/>')
            if start_pt is None and idx_s == 0:
                start_pt = (ox + (s_geom.coords[0][0] - minx) * scale, oy + ph - (s_geom.coords[0][1] - miny) * scale)
            end_pt = (ox + (s_geom.coords[-1][0] - minx) * scale, oy + ph - (s_geom.coords[-1][1] - miny) * scale)

    # Origin & Destination markers
    if start_pt:
        svg.append(f'<circle cx="{start_pt[0]:.1f}" cy="{start_pt[1]:.1f}" r="8" fill="#16a34a" stroke="#ffffff" stroke-width="2"/>')
        svg.append(f'<text x="{start_pt[0]:.1f}" y="{start_pt[1] + 3:.1f}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9" font-weight="bold" fill="#ffffff">S</text>')
    if end_pt:
        svg.append(f'<circle cx="{end_pt[0]:.1f}" cy="{end_pt[1]:.1f}" r="8" fill="#dc2626" stroke="#ffffff" stroke-width="2"/>')
        svg.append(f'<text x="{end_pt[0]:.1f}" y="{end_pt[1] + 3:.1f}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9" font-weight="bold" fill="#ffffff">D</text>')

    # Legend (Clean white card)
    svg.append(f'<rect x="{ox + 15}" y="{oy + 15}" width="310" height="110" fill="{BG_WHITE}" stroke="#cbd5e1" stroke-width="1.2" rx="4" opacity="0.96"/>')
    leg_items = [
        ('Shortest Path (High Exposure: 951 m inundated)', '#dc2626', '3.2'),
        ('ML Flood-Aware Route (Detour: 108 m inundated)', '#2563eb', '3.2'),
        ('Oracle Optimal Route (Ground Truth: 0 m inundated)', '#16a34a', '3.2'),
        ('Background Flooded Segments (>30 cm)', '#fecaca', '2.0')
    ]
    for idx, (label, col, lw_leg) in enumerate(leg_items):
        ly = oy + 34 + idx * 22
        svg.append(f'<line x1="{ox + 28}" y1="{ly}" x2="{ox + 55}" y2="{ly}" stroke="{col}" stroke-width="{lw_leg}"/>')
        svg.append(f'<text x="{ox + 65}" y="{ly + 4}" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_SUBTITLE}">{label}</text>')

    # North Arrow
    nx, ny = ox + pw - 30, oy + 35
    svg.append(f'<polygon points="{nx},{ny} {nx-5},{ny+14} {nx+5},{ny+14}" fill="{TEXT_TITLE}"/>')
    svg.append(f'<text x="{nx}" y="{ny-4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" font-weight="bold" fill="{TEXT_TITLE}">N</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")


def generate_fig10_sensitivity(block_data, label_data, out_path_svg):
    """Fig. 10: Sensitivity of classification and labeling to block size & uncertain labels."""
    w, h = 940, 450
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append(f'<rect width="100%" height="100%" fill="{BG_WHITE}"/>')

    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="{FONT_STACK}" font-size="14" font-weight="bold" fill="{TEXT_TITLE}">Sensitivity Analysis of Spatial Validation Settings and Target Labeling</text>')

    # Panel A: Spatial Block Size (x: 65..445, y: 60..360)
    ox_a, oy_a, pw_a, ph_a = 65, 60, 380, 300
    svg.append(f'<text x="{ox_a + pw_a/2}" y="{oy_a - 12}" text-anchor="middle" font-family="{FONT_STACK}" font-size="12" font-weight="bold" fill="{TEXT_SUBTITLE}">(a) Spatial Block Dimension (m)</text>')

    # Y-axis ticks for Panel A (0.7 to 1.0)
    for tick in np.linspace(0.7, 1.0, 7):
        ty = oy_a + (1.0 - (tick - 0.7) / 0.3) * ph_a
        svg.append(f'<line x1="{ox_a}" y1="{ty}" x2="{ox_a + pw_a}" y2="{ty}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{ox_a - 8}" y="{ty + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{tick:.2f}</text>')

    svg.append(f'<rect x="{ox_a}" y="{oy_a}" width="{pw_a}" height="{ph_a}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')
    svg.append(f'<text x="{ox_a - 45}" y="{oy_a + ph_a/2}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}" transform="rotate(-90 {ox_a-45} {oy_a+ph_a/2})">Validation Metric Score</text>')

    b_sizes = list(block_data.keys())
    b_w = pw_a / len(b_sizes)
    for i, bs in enumerate(b_sizes):
        bx = ox_a + i * b_w + b_w * 0.18
        val_pr = block_data[bs]['auc_pr']
        val_roc = block_data[bs]['auc_roc']
        h_pr = max(0, ((val_pr - 0.7) / 0.3) * ph_a)
        h_roc = max(0, ((val_roc - 0.7) / 0.3) * ph_a)

        # PR bar
        svg.append(f'<rect x="{bx}" y="{oy_a + ph_a - h_pr}" width="{b_w*0.3}" height="{h_pr}" fill="#2563eb" rx="2"/>')
        svg.append(f'<text x="{bx + b_w*0.15}" y="{oy_a + ph_a - h_pr - 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#1e40af">{val_pr:.3f}</text>')

        # ROC bar
        svg.append(f'<rect x="{bx + b_w*0.3 + 4}" y="{oy_a + ph_a - h_roc}" width="{b_w*0.3}" height="{h_roc}" fill="#059669" rx="2"/>')
        svg.append(f'<text x="{bx + b_w*0.45 + 4}" y="{oy_a + ph_a - h_roc - 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#065f46">{val_roc:.3f}</text>')

        svg.append(f'<text x="{bx + b_w*0.3 + 2}" y="{oy_a + ph_a + 18}" text-anchor="middle" font-family="{FONT_STACK}" font-size="11" font-weight="600" fill="{TEXT_SUBTITLE}">{bs}</text>')

    # Panel B: Uncertain Label Handling (x: 495..875, y: 60..360)
    ox_b, oy_b, pw_b, ph_b = 495, 60, 380, 300
    svg.append(f'<text x="{ox_b + pw_b/2}" y="{oy_b - 12}" text-anchor="middle" font-family="{FONT_STACK}" font-size="12" font-weight="bold" fill="{TEXT_SUBTITLE}">(b) Uncertain Segment Strategy</text>')

    for tick in np.linspace(0.7, 1.0, 7):
        ty = oy_b + (1.0 - (tick - 0.7) / 0.3) * ph_b
        svg.append(f'<line x1="{ox_b}" y1="{ty}" x2="{ox_b + pw_b}" y2="{ty}" stroke="{GRID_LINE}" stroke-width="1"/>')
        svg.append(f'<text x="{ox_b - 8}" y="{ty + 4}" text-anchor="end" font-family="{FONT_STACK}" font-size="10" fill="{TEXT_MUTED}">{tick:.2f}</text>')

    svg.append(f'<rect x="{ox_b}" y="{oy_b}" width="{pw_b}" height="{ph_b}" fill="none" stroke="{BORDER_AXIS}" stroke-width="1.2"/>')

    l_types = list(label_data.keys())
    l_w = pw_b / len(l_types)
    for i, lt in enumerate(l_types):
        lx = ox_b + i * l_w + l_w * 0.15
        val_pr = label_data[lt]['auc_pr']
        val_roc = label_data[lt]['auc_roc']
        h_pr = max(0, ((val_pr - 0.7) / 0.3) * ph_b)
        h_roc = max(0, ((val_roc - 0.7) / 0.3) * ph_b)

        svg.append(f'<rect x="{lx}" y="{oy_b + ph_b - h_pr}" width="{l_w*0.33}" height="{h_pr}" fill="#2563eb" rx="2"/>')
        svg.append(f'<text x="{lx + l_w*0.165}" y="{oy_b + ph_b - h_pr - 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#1e40af">{val_pr:.3f}</text>')

        svg.append(f'<rect x="{lx + l_w*0.33 + 4}" y="{oy_b + ph_b - h_roc}" width="{l_w*0.33}" height="{h_roc}" fill="#059669" rx="2"/>')
        svg.append(f'<text x="{lx + l_w*0.495 + 4}" y="{oy_b + ph_b - h_roc - 4}" text-anchor="middle" font-family="{FONT_STACK}" font-size="9.5" font-weight="bold" fill="#065f46">{val_roc:.3f}</text>')

        # Format label name cleanly
        disp_lt = lt.replace(" (main)", "\n(main)").split("\n")[0]
        svg.append(f'<text x="{lx + l_w*0.33 + 2}" y="{oy_b + ph_a + 18}" text-anchor="middle" font-family="{FONT_STACK}" font-size="10" font-weight="600" fill="{TEXT_SUBTITLE}">{disp_lt}</text>')

    # Legend at bottom center
    ly = oy_a + ph_a + 42
    svg.append(f'<rect x="{w/2 - 140}" y="{ly}" width="16" height="12" fill="#2563eb" rx="2"/>')
    svg.append(f'<text x="{w/2 - 116}" y="{ly + 10}" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_SUBTITLE}">Spatial Block AUC-PR</text>')
    svg.append(f'<rect x="{w/2 + 50}" y="{ly}" width="16" height="12" fill="#059669" rx="2"/>')
    svg.append(f'<text x="{w/2 + 74}" y="{ly + 10}" font-family="{FONT_STACK}" font-size="11" fill="{TEXT_SUBTITLE}">Spatial Block AUC-ROC</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")
