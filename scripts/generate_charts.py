"""
Dedicated visualization generator using SVG and Pillow for publication-grade charts.
Produces vector SVG and high-resolution PNG for:
- Fig 1: ROC & Precision-Recall curves across 6 models
- Fig 2: Optimism Gap grouped bar chart
- Fig 3: TreeSHAP Feature Importance horizontal bar chart
- Fig 4: Downstream Routing Trade-off scatter plot
- Fig 5: Road Network Flood Risk Vector Map
"""
import os
import math
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path

OUT_DIR = str(Path(__file__).resolve().parent.parent / 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

def generate_fig1_roc_pr(roc_pr_data, out_path_svg):
    """Docstring"""
    w, h = 1000, 480
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    
    # Left panel: ROC (x: 80..440, y: 60..400)
    # Right panel: PR (x: 580..940, y: 60..400)
    panels = [
        {'ox': 80, 'oy': 60, 'pw': 360, 'ph': 340, 'title': 'Spatial Block CV: ROC Curves', 'xlabel': 'False Positive Rate', 'ylabel': 'True Positive Rate'},
        {'ox': 580, 'oy': 60, 'pw': 360, 'ph': 340, 'title': 'Spatial Block CV: Precision-Recall Curves', 'xlabel': 'Recall (Flooded Class)', 'ylabel': 'Precision'}
    ]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for p in panels:
        ox, oy, pw, ph = p['ox'], p['oy'], p['pw'], p['ph']
        svg.append(f'<text x="{ox + pw/2}" y="{oy - 20}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="15" font-weight="bold" fill="#222">{p["title"]}</text>')
        # Grid lines
        for tick in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            tx = ox + tick * pw
            ty = oy + (1.0 - tick) * ph
            svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="#e5e7eb" stroke-width="1"/>')
            svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="#e5e7eb" stroke-width="1"/>')
            svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="11" fill="#555">{tick:.1f}</text>')
            svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="11" fill="#555">{tick:.1f}</text>')
        
        svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="#6b7280" stroke-width="1.2"/>')
        svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 42}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="12" font-weight="600" fill="#374151">{p["xlabel"]}</text>')
        svg.append(f'<text x="{ox - 45}" y="{oy + ph/2}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="12" font-weight="600" fill="#374151" transform="rotate(-90 {ox-45} {oy+ph/2})">{p["ylabel"]}</text>')

    # Diagonal reference on ROC
    svg.append(f'<line x1="{80}" y1="{60+340}" x2="{80+360}" y2="{60}" stroke="#9ca3af" stroke-dasharray="4,4" stroke-width="1.2"/>')
    # Baseline on PR (0.425)
    baseline_y = 60 + (1.0 - 0.425) * 340
    svg.append(f'<line x1="{580}" y1="{baseline_y}" x2="{580+360}" y2="{baseline_y}" stroke="#9ca3af" stroke-dasharray="4,4" stroke-width="1.2"/>')

    # Draw curves
    for idx, (m_name, m_data) in enumerate(roc_pr_data.items()):
        col = colors[idx]
        # ROC
        fpr, tpr = m_data['fpr'], m_data['tpr']
        pts_roc = [f"{80 + x*360:.1f},{60 + (1-y)*340:.1f}" for x, y in zip(fpr, tpr)]
        svg.append(f'<polyline points="{" ".join(pts_roc)}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        
        # PR
        rec, prec = m_data['rec'], m_data['prec']
        pts_pr = [f"{580 + x*360:.1f},{60 + (1-y)*340:.1f}" for x, y in zip(rec, prec)]
        svg.append(f'<polyline points="{" ".join(pts_pr)}" fill="none" stroke="{col}" stroke-width="2.2"/>')
        
        # Legend items
        ly = 260 + idx * 20
        svg.append(f'<line x1="{240}" y1="{ly}" x2="{260}" y2="{ly}" stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<text x="{268}" y="{ly + 4}" font-family="Segoe UI, sans-serif" font-size="11" fill="#222">{m_name} (AUC={m_data["auc_roc"]:.3f})</text>')
        
        svg.append(f'<line x1="{600}" y1="{ly}" x2="{620}" y2="{ly}" stroke="{col}" stroke-width="2.5"/>')
        svg.append(f'<text x="{628}" y="{ly + 4}" font-family="Segoe UI, sans-serif" font-size="11" fill="#222">{m_name} (AUC-PR={m_data["auc_pr"]:.3f})</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

def generate_fig2_optimism_gap(models, spatial_aucs, random_aucs, out_path_svg):
    """Grouped bar chart for Spatial vs Random CV."""
    w, h = 760, 420
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    ox, oy, pw, ph = 70, 60, 640, 280
    
    svg.append(f'<text x="{ox + pw/2}" y="32" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="15" font-weight="bold" fill="#222">Optimism Gap: Random Cross-Validation Overstatement vs Spatial Block CV</text>')
    
    # Y-axis ticks (0.7 to 1.0)
    for tick in np.linspace(0.7, 1.0, 7):
        ty = oy + (1.0 - (tick - 0.7) / 0.3) * ph
        svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="#f0f0f0" stroke-width="1"/>')
        svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="11" fill="#555">{tick:.2f}</text>')

    svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="#6b7280" stroke-width="1"/>')
    svg.append(f'<text x="{ox - 42}" y="{oy + ph/2}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="12" font-weight="600" fill="#374151" transform="rotate(-90 {ox-42} {oy+ph/2})">AUC-ROC</text>')

    n = len(models)
    group_w = pw / n
    bar_w = group_w * 0.35
    
    for i, m in enumerate(models):
        gx = ox + i * group_w + group_w * 0.15
        sp_h = ((spatial_aucs[i] - 0.7) / 0.3) * ph
        rd_h = ((random_aucs[i] - 0.7) / 0.3) * ph
        
        sp_y = oy + ph - sp_h
        rd_y = oy + ph - rd_h
        
        # Spatial Bar
        svg.append(f'<rect x="{gx}" y="{sp_y}" width="{bar_w}" height="{sp_h}" fill="#1f77b4" rx="2"/>')
        svg.append(f'<text x="{gx + bar_w/2}" y="{sp_y - 5}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="10" font-weight="600" fill="#1f77b4">{spatial_aucs[i]:.3f}</text>')
        
        # Random Bar
        svg.append(f'<rect x="{gx + bar_w + 4}" y="{rd_y}" width="{bar_w}" height="{rd_h}" fill="#ff7f0e" rx="2"/>')
        svg.append(f'<text x="{gx + bar_w*1.5 + 4}" y="{rd_y - 5}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="10" font-weight="600" fill="#d95f02">{random_aucs[i]:.3f}</text>')
        
        # Gap label
        gap = random_aucs[i] - spatial_aucs[i]
        svg.append(f'<text x="{gx + bar_w + 2}" y="{oy + ph + 34}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="10" font-weight="bold" fill="#b91c1c">Δ={gap:+.3f}</text>')
        svg.append(f'<text x="{gx + bar_w + 2}" y="{oy + ph + 18}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="11" font-weight="600" fill="#333">{m}</text>')

    # Legend
    svg.append(f'<rect x="{w - 260}" y="{oy + 10}" width="16" height="12" fill="#1f77b4" rx="2"/>')
    svg.append(f'<text x="{w - 235}" y="{oy + 21}" font-family="Segoe UI, sans-serif" font-size="11" fill="#333">Spatial Block CV</text>')
    svg.append(f'<rect x="{w - 260}" y="{oy + 30}" width="16" height="12" fill="#ff7f0e" rx="2"/>')
    svg.append(f'<text x="{w - 235}" y="{oy + 41}" font-family="Segoe UI, sans-serif" font-size="11" fill="#333">Random CV (Leaked)</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

def generate_fig3_shap(feat_names, shap_vals, out_path_svg):
    """Horizontal bar chart of SHAP values."""
    w, h = 680, 420
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    ox, oy, pw, ph = 160, 50, 480, 320
    
    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="14" font-weight="bold" fill="#222">Global Feature Importance via TreeSHAP (XGBoost)</text>')
    
    max_val = max(shap_vals) * 1.15
    # Ticks
    for tick in np.linspace(0, max_val, 6):
        tx = ox + (tick / max_val) * pw
        svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="#f3f4f6" stroke-width="1"/>')
        svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="10" fill="#555">{tick:.2f}</text>')

    svg.append(f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy + ph}" stroke="#4b5563" stroke-width="1.2"/>')
    svg.append(f'<line x1="{ox}" y1="{oy + ph}" x2="{ox + pw}" y2="{oy + ph}" stroke="#4b5563" stroke-width="1.2"/>')
    svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 38}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="11" font-weight="600" fill="#374151">Mean |SHAP Value| (Impact on Model Log-Odds)</text>')

    n = len(feat_names)
    bar_h = ph / (n * 1.5)
    for i in range(n):
        val = shap_vals[i]
        bw = (val / max_val) * pw
        by = oy + i * (ph / n) + 6
        svg.append(f'<rect x="{ox}" y="{by}" width="{bw}" height="{bar_h}" fill="#2ca02c" rx="3"/>')
        svg.append(f'<text x="{ox - 8}" y="{by + bar_h/2 + 4}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="11" font-weight="600" fill="#333">{feat_names[i]}</text>')
        svg.append(f'<text x="{ox + bw + 6}" y="{by + bar_h/2 + 4}" font-family="Segoe UI, sans-serif" font-size="10" font-weight="bold" fill="#2ca02c">{val:.3f}</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

def generate_fig4_routing(strategies, times, floods, out_path_svg):
    """Scatter plot of routing trade-offs."""
    w, h = 760, 460
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
    ox, oy, pw, ph = 80, 50, 620, 350
    
    svg.append(f'<text x="{w/2}" y="28" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="14" font-weight="bold" fill="#222">Downstream Routing Benchmark: Hazard Avoidance vs Travel-Time Overhead</text>')
    
    # X: 0..25%, Y: 400..1150m
    max_t = 25.0
    min_fl, max_fl = 400.0, 1150.0
    
    # Grid lines
    for t_tick in [0, 5, 10, 15, 20, 25]:
        tx = ox + (t_tick / max_t) * pw
        svg.append(f'<line x1="{tx}" y1="{oy}" x2="{tx}" y2="{oy + ph}" stroke="#f3f4f6" stroke-width="1"/>')
        svg.append(f'<text x="{tx}" y="{oy + ph + 18}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="10" fill="#555">{t_tick}%</text>')

    for fl_tick in [400, 600, 800, 1000]:
        ty = oy + ph - ((fl_tick - min_fl) / (max_fl - min_fl)) * ph
        svg.append(f'<line x1="{ox}" y1="{ty}" x2="{ox + pw}" y2="{ty}" stroke="#f3f4f6" stroke-width="1"/>')
        svg.append(f'<text x="{ox - 8}" y="{ty + 4}" text-anchor="end" font-family="Segoe UI, sans-serif" font-size="10" fill="#555">{fl_tick}m</text>')

    svg.append(f'<rect x="{ox}" y="{oy}" width="{pw}" height="{ph}" fill="none" stroke="#6b7280" stroke-width="1"/>')
    svg.append(f'<text x="{ox + pw/2}" y="{oy + ph + 38}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="11" font-weight="600" fill="#374151">Median Travel Time Increase Over Shortest Path (%)</text>')
    svg.append(f'<text x="{ox - 50}" y="{oy + ph/2}" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="11" font-weight="600" fill="#374151" transform="rotate(-90 {ox-50} {oy+ph/2})">Flooded Distance per Trip (m)</text>')

    strat_colors = {
        'Shortest time': '#d62728',
        'LR': '#1f77b4',
        'SVM (RBF)': '#9467bd',
        'Random Forest': '#8c564b',
        'XGBoost': '#e377c2',
        'MLP': '#17becf',
        'GraphSAGE': '#bcbd22',
        'Oracle (observed)': '#2ca02c'
    }

    for i, s in enumerate(strategies):
        t_val = times[i]
        fl_val = floods[i]
        cx = ox + (t_val / max_t) * pw
        cy = oy + ph - ((fl_val - min_fl) / (max_fl - min_fl)) * ph
        col = strat_colors.get(s, '#333333')
        
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="7" fill="{col}" stroke="#111827" stroke-width="1.5"/>')
        offset_y = -12 if i % 2 == 0 else 16
        offset_x = 8 if cx < ox + pw - 90 else -80
        svg.append(f'<text x="{cx + offset_x}" y="{cy + offset_y}" font-family="Segoe UI, sans-serif" font-size="10.5" font-weight="bold" fill="{col}">{s} ({fl_val:.0f}m, +{t_val:.1f}%)</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

def generate_fig5_map(geoms_wkt, probs, out_path_svg):
    """Vector road network risk map."""
    import shapely.wkt as wkt
    lines = [wkt.loads(g) for g in geoms_wkt]
    all_coords = [pt for l in lines for pt in l.coords]
    xs = [pt[0] for pt in all_coords]
    ys = [pt[1] for pt in all_coords]
    
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    
    w, h = 640, 860
    ox, oy, pw, ph = 40, 60, 560, 740
    scale = min(pw / (maxx - minx), ph / (maxy - miny))
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    svg.append('<rect width="100%" height="100%" fill="#111827"/>')
    svg.append(f'<text x="{w/2}" y="36" text-anchor="middle" font-family="Segoe UI, sans-serif" font-size="15" font-weight="bold" fill="#ffffff">Da Nang (Hai Chau): Out-of-Fold Road Flood Risk Map</text>')
    
    for line, p in zip(lines, probs):
        if p >= 0.66:
            col = '#ef4444' # High risk red
            lw = 2.0
        elif p >= 0.33:
            col = '#f59e0b' # Medium risk orange
            lw = 1.3
        else:
            col = '#3b82f6' # Low risk blue
            lw = 0.9
            
        pts = []
        for x, y in line.coords:
            sx = ox + (x - minx) * scale
            sy = oy + ph - (y - miny) * scale
            pts.append(f"{sx:.1f},{sy:.1f}")
        svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{col}" stroke-width="{lw}" stroke-linecap="round" opacity="0.85"/>')

    # Legend
    svg.append(f'<rect x="{ox + 10}" y="{oy + 10}" width="180" height="90" fill="#1f2937" rx="6" opacity="0.9"/>')
    legend_tiers = [
        ('Low Risk (p &lt; 0.33)', '#3b82f6'),
        ('Medium Risk (0.33 - 0.66)', '#f59e0b'),
        ('High Risk (p &gt; 0.66)', '#ef4444')
    ]
    for idx, (label, col) in enumerate(legend_tiers):
        ly = oy + 32 + idx * 24
        svg.append(f'<line x1="{ox + 25}" y1="{ly}" x2="{ox + 50}" y2="{ly}" stroke="{col}" stroke-width="3"/>')
        svg.append(f'<text x="{ox + 60}" y="{ly + 4}" font-family="Segoe UI, sans-serif" font-size="11" fill="#e5e7eb">{label}</text>')

    svg.append('</svg>')
    with open(out_path_svg, 'w', encoding='utf-8') as f:
        f.write('\n'.join(svg))
    print(f"Generated: {out_path_svg}")

print("Visualization generator module initialized.")
