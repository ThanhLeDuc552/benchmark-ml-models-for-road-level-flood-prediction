# Benchmarking Machine Learning Models for Road-Level Flood Prediction and Flood-Aware Routing

### *A Case Study of Hai Chau District, Da Nang, Vietnam (14–15 October 2022 Extreme Flood Event)*

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C.svg)](https://pyg.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-red.svg)](https://xgboost.readthedocs.io/)
[![HEC-RAS 2D](https://img.shields.io/badge/Ground_Truth-HEC--RAS_2D-0077B6.svg)](https://www.hec.usace.army.mil/software/hec-ras/)
[![CRS EPSG:2045](https://img.shields.io/badge/CRS-EPSG%3A2045-green.svg)](https://epsg.io/2045)

---

## 1. Overview & Research Motivation

Urban flooding driven by extreme precipitation poses critical hazards to transport networks, emergency response, and municipal mobility. Standard machine learning workflows applied to geospatial infrastructure often suffer from two major flaws:
1. **Spatial Autocorrelation Leakage**: Standard random cross-validation inflates performance metrics because spatially contiguous segments leak local hydraulic signatures across folds, creating an **Optimism Gap** ($\Delta_m = \text{AUC}_m^{\text{random}} - \text{AUC}_m^{\text{spatial}}$).
2. **Classification vs. Routing Disconnect**: Model rankings on classical classification metrics (e.g., AUC-PR, F1 score) do not necessarily predict downstream routing efficacy when probabilities are integrated into dynamic routing costs ($C_i = t_i (1 + \lambda p_i)$).

This repository implements a rigorous, end-to-end benchmark protocol mirroring the accompanying research paper (**[IEEE_Conference_Template.pdf](IEEE_Conference_Template.pdf)**) for Hai Chau District, Da Nang, Vietnam, capturing the historic 14–15 October 2022 pluvial flood.

```
                           [ Raw Geospatial & Hydraulic Inputs ]
                     (HEC-RAS 2D, Copernicus DEM, OSM, HEC-DSS Rain)
                                          │
                                          ▼
                         [ Modular Feature Engineering Pipeline ]
                   (Hydrology, Topography, Network GraphML, DSS Gauge)
                                          │
                                          ▼
                      [ Leakage-Aware Spatial Block CV (500m) ]
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
         [ 6 ML Models / 5 Families ]               [ Model Explainability ]
       (LR, SVM, RF, XGB, MLP, SAGE)                       (TreeSHAP)
                     │
                     ▼
       [ Downstream Flood-Aware Routing ]
        (Cost: C_i = t_i * (1 + λ * p_i))
```

---

## 2. Repository Structure

```
.
├── IEEE_Conference_Template.pdf     # Reference conference research paper
├── README.md                        # Primary project documentation
├── dataset/                         # Standardized benchmark datasets
│   ├── feature_table.xlsx           # Segment-level features and ground truth passability
│   ├── hai_chau_road_graph.graphml  # NetworkX/Dijkstra road graph with encoded attributes
│   └── hai_chau_road_graph.gpkg     # Projected spatial vector layer (EPSG:2045)
├── figures/                         # Publication-grade figures and evaluation summaries
│   ├── evaluation.md                # Quantitative classification metrics & optimism gap
│   ├── fig1_roc_pr_curves.svg       # Spatial Block CV ROC & Precision-Recall curves
│   ├── fig2_optimism_gap.svg        # Grouped bar chart: Spatial Block CV vs Random CV
│   ├── fig3_shap_importance.svg     # TreeSHAP global feature attribution (XGBoost)
│   ├── fig4_routing_tradeoff.svg    # Pareto scatter: Flooded distance vs Travel time overhead
│   └── fig5_risk_map.svg            # District-wide road inundation risk map (EPSG:2045)
├── new DSSs/                        # Gauge time-series data
│   ├── stations.csv                 # 428 rain gauge coordinates
│   └── station_0000/ ...            # HEC-DSS precipitation record archives
├── resources for forming dataset/   # Raw hydraulic, elevation, and administrative layers
│   ├── hec_ras_run_2022-10-14_to_2022-10-15/  # Unsteady 2D simulation HDF (p05.hdf)
│   ├── danangfloodmodel.g01.hdf     # Geometry Centerlines and 2D flow areas
│   ├── COPERNICUS_DANANG_FULL_EPSG2045_clean.tif  # 30m Digital Elevation Model
│   ├── depression_depth.tif         # Hydrological sinks and depression depths
│   ├── tpi.tif                      # Topographic Position Index raster
│   ├── danang.gpkg                  # Regional base spatial layers
│   └── hai_chau.geojson             # Hai Chau district administrative boundary
└── scripts/                         # Pipelines, models, notebooks, and modules
    ├── road_level_flood_pred.ipynb  # Self-contained master Jupyter notebook
    ├── run_full_benchmark.py        # CLI benchmark execution & evaluation script
    ├── generate_charts.py           # Publication-grade SVG/raster visualization generator
    ├── build_notebook.py            # Reproducible notebook generator using nbformat
    ├── populate_notebook_outputs.py # Headless notebook execution & output populator
    ├── scratch_benchmark_test.py    # Rapid smoke test for data and pipeline integrity
    ├── build_phase1_features.py     # Master orchestrator for feature extraction
    ├── ground_truth/                # HEC-RAS 2D depth extraction & passability derivation
    ├── hydrology_data/              # River centerline extraction & distance calculations
    ├── rainfall_data/               # HEC-DSS extraction & nearest-station spatial join
    ├── road_data/                   # Road network extraction, segment splitting (<=100m)
    └── terrain_data/                # DEM zonal statistics & Horn's slope derivation
```

---

## 3. Study Area & Dataset Specifications

All vector layers and raster grids are standardized in **EPSG:2045** (Hanoi 1972 / GK Zone 19).

* **Study Area**: Hai Chau district, Da Nang, Vietnam ($24.08\text{ km}^2$, dense urban delta).
* **Road Network**: Extracted from OpenStreetMap, segmented into sub-edges of length $\le 100\text{ m}$.
  * **Intersections (Nodes)**: $1,190$
  * **Road Segments (Edges)**: $2,059$
  * **Dual Line Graph ($L(G)$)**: $2,059$ nodes, $13,594$ directed adjacency edges.
* **Ground Truth Target ($y_i$)**: Derived from HEC-RAS 2D unsteady hydraulic simulation for 14–15 October 2022:
  * `passability = 0`: Passable ($0 - 10\text{ cm}$ depth, $n=1,183$, $57.5\%$)
  * `passability = 1`: Flooded but passable ($10 - 30\text{ cm}$ depth, $n=323$, $15.7\%$)
  * `passability = 2` (**Impassable Target $y=1$**): $> 30\text{ cm}$ water depth ($n=553$, $26.9\%$). Under primary binary formulation ($y \in \{0, 1\}$), impassable segments represent $42.5\%$ of active transport edges.

### Feature Dictionary

| Attribute | Category | Description | Data Type / Unit |
| :--- | :--- | :--- | :--- |
| `segment_id` | Identifier | Unique identifier for road segment edge | String |
| `length_m` | Network | Edge length (split at $\le 100\text{ m}$) | Meters |
| `node_degree` | Network | Structural connectivity degree of start intersection | Count |
| `road_class` | Network | OSM classification (primary, secondary, residential, etc.) | One-Hot (7 categories) |
| `is_underpass_tunnel_bridge` | Network | Structural indicator for tunnels, bridges, underpasses | Binary (0 / 1) |
| `betweenness` | Network | Normalized edge betweenness centrality | Float $[0, 1]$ |
| `dist_river` | Hydrology | Euclidean distance to Han River centerline | Meters |
| `elev_min` | Topography | Minimum DEM elevation in 50 m segment buffer | Meters |
| `elev_mean` | Topography | Mean DEM elevation in 50 m segment buffer | Meters |
| `slope_mean` | Topography | Mean Horn's slope within 50 m buffer | Degrees |
| `tpi` | Topography | Mean Topographic Position Index in 50 m buffer | Meters |
| `depression_depth` | Topography | Maximum sink/depression depth in 50 m buffer | Meters |
| `rain_total` | Rainfall | Event total accumulated precipitation | mm |
| `rain_peak_intensity`| Rainfall | Peak hourly precipitation intensity | mm/h |

---

## 4. Benchmark Methodology

### 1. Spatial Block Cross-Validation
To prevent spatial data leakage, Hai Chau district is discretized into contiguous $500\text{ m} \times 500\text{ m}$ square spatial blocks assigned to 5 spatial folds. All models are trained and tested on non-overlapping spatial folds and benchmarked against standard 5-fold Random CV to quantify the **Optimism Gap**:

$$\Delta_m = \text{AUC-ROC}_m^{\text{random}} - \text{AUC-ROC}_m^{\text{spatial}}$$

### 2. Six Benchmark Models across Five Families
1. **Linear**: Logistic Regression (`LR`) with $L_2$ regularization and balanced class weighting.
2. **Kernel**: Support Vector Machine with Radial Basis Function (`SVM (RBF)`).
3. **Tree Ensembles**: Random Forest (`RF`, 300 estimators) and Extreme Gradient Boosting (`XGBoost`, max_depth=5, lr=0.05).
4. **Neural**: Multilayer Perceptron (`MLP`, 2 hidden layers: 128-64, ReLU, Dropout=0.3).
5. **Graph Neural Network**: `GraphSAGE` (2-layer inductive neighborhood aggregation on the dual line graph $L(G)$).

### 3. Downstream Flood-Aware Routing
Out-of-fold hazard probabilities $p_i \in [0, 1]$ are mapped to modified travel time costs:

$$C_i = t_i \cdot (1 + \lambda \cdot p_i)$$

where $t_i = \text{length}_i / v_i$ is free-flow travel time, and $\lambda = 10.0$ penalizes flood hazard exposure. Routing is benchmarked over 100 randomly sampled origin-destination (OD) pairs against:
* **Shortest Time Baseline**: Ignores flood hazard ($\lambda = 0$).
* **Oracle Baseline**: Routes with perfect observed ground truth ($p_i = y_i$).

---

## 5. Experimental Results

### Table I: Model Performance under Spatial Block CV vs Random CV

| Model | Family | AUC-PR ↑ | AUC-ROC ↑ | F1 Score ↑ | Recall (Flooded) ↑ | Brier Score ↓ | Random CV AUC-ROC | Optimism Gap ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LR** | Linear | $0.852 \pm 0.052$ | $0.873 \pm 0.040$ | $0.789 \pm 0.066$ | $0.821 \pm 0.067$ | $0.143 \pm 0.029$ | $0.897 \pm 0.017$ | $+0.024$ |
| **SVM (RBF)** | Kernel | $0.863 \pm 0.050$ | $0.900 \pm 0.029$ | $0.800 \pm 0.037$ | $0.819 \pm 0.043$ | $0.126 \pm 0.024$ | $0.936 \pm 0.009$ | $+0.036$ |
| **Random Forest**| Tree Ensemble | $0.880 \pm 0.051$ | $0.904 \pm 0.032$ | $0.788 \pm 0.034$ | $0.811 \pm 0.049$ | $0.126 \pm 0.027$ | $0.992 \pm 0.005$ | **$+0.088$** |
| **XGBoost** | Tree Ensemble | $0.891 \pm 0.035$ | $0.909 \pm 0.024$ | $0.805 \pm 0.034$ | $0.838 \pm 0.069$ | $0.132 \pm 0.024$ | $0.978 \pm 0.007$ | **$+0.070$** |
| **MLP** | Neural | **$0.889 \pm 0.056$** | $0.913 \pm 0.030$ | $0.824 \pm 0.043$ | **$0.870 \pm 0.032$** | **$0.124 \pm 0.029$** | $0.948 \pm 0.011$ | $+0.035$ |
| **GraphSAGE** | Graph Neural | $0.879 \pm 0.055$ | **$0.914 \pm 0.024$** | **$0.827 \pm 0.039$** | $0.862 \pm 0.042$ | $0.127 \pm 0.029$ | $0.969 \pm 0.011$ | $+0.055$ |

### Table II: Downstream Routing Benchmark (100 OD Pairs, $\lambda = 10$)

| Strategy | Flooded Segments / Trip | Flooded Length (m) | Time Increase (Median, %) | No Safe Route (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Shortest Time (Unaware)** | 6.42 | 1063.6 m | 0.0% | 0.0% |
| **Oracle (Observed Truth)** | **0.00** | **0.0 m** | +8.9% | 0.0% |
| **Logistic Regression** | 4.28 | 719.4 m | +12.3% | 0.0% |
| **SVM (RBF)** | 4.10 | 688.5 m | +13.5% | 0.0% |
| **Random Forest** | 3.82 | 642.1 m | +14.1% | 0.0% |
| **XGBoost (Best Routing)** | **3.51** | **594.4 m** | **+14.8%** | **0.0%** |
| **MLP** | 3.65 | 618.2 m | +15.2% | 0.0% |
| **GraphSAGE** | 3.60 | 609.8 m | +15.0% | 0.0% |

### Core Scientific Findings
1. **The Classification vs Routing Disconnect**: While **MLP** achieved the highest classification precision and recall ($87.0\%$), **XGBoost** produced the superior downstream navigation outcome, reducing flooded travel distance from $1,063.6\text{ m}$ to $594.4\text{ m}$ (a **$44.1\%$ reduction**) with only a $14.8\%$ travel time increase.
2. **Severe Optimism Gap in Tree Models**: Tree ensembles showed the greatest vulnerability to spatial autocorrelation leakage ($+0.088$ in RF and $+0.070$ in XGBoost), showing near-perfect performance under random CV that drops significantly under spatial CV.
3. **Key Morphological Drivers**: TreeSHAP analysis identified `depression_depth` and `tpi` (Topographic Position Index) as the primary drivers of inundation risk, followed by `dist_river` and `elev_mean`.

---

## 6. Publication Figures

All vector figures are stored in [`figures/`](figures/):

| Figure | Description | File |
| :---: | :--- | :--- |
| **Fig. 1** | Spatial Block CV: ROC and Precision-Recall Curves across all 6 models | [`figures/fig1_roc_pr_curves.svg`](figures/fig1_roc_pr_curves.svg) |
| **Fig. 2** | Optimism Gap comparison (Random CV vs. Spatial Block CV) | [`figures/fig2_optimism_gap.svg`](figures/fig2_optimism_gap.svg) |
| **Fig. 3** | Global TreeSHAP feature importance ranking (XGBoost) | [`figures/fig3_shap_importance.svg`](figures/fig3_shap_importance.svg) |
| **Fig. 4** | Downstream routing Pareto trade-off (Flooded Distance vs. Time Increase) | [`figures/fig4_routing_tradeoff.svg`](figures/fig4_routing_tradeoff.svg) |
| **Fig. 5** | High-resolution road segment inundation risk vector map (Hai Chau) | [`figures/fig5_risk_map.svg`](figures/fig5_risk_map.svg) |

---

## 7. Quickstart & Installation

### Environment Setup
Create and activate an environment with Python 3.10 and GIS dependencies:

```bash
# Using Conda / Mamba
conda create -n flood-gis python=3.10 -y
conda activate flood-gis

# Install geospatial & ML dependencies
conda install -c conda-forge gdal proj geopandas pyogrio shapely networkx openpyxl -y
pip install torch torchvision torchaudio
pip install torch_geometric
pip install xgboost scikit-learn seaborn matplotlib pillow nbformat
```

### Reproducing the Benchmark
To run the full benchmark pipeline and regenerate all metrics, statistical tests, routing tables, and figures:

```bash
# 1. Run quick verification test (smoke test)
python scripts/scratch_benchmark_test.py

# 2. Run the end-to-end benchmark CLI script
python scripts/run_full_benchmark.py

# 3. Open the interactive Jupyter notebook
jupyter notebook scripts/road_level_flood_pred.ipynb
```

---

## 8. Citation

If you utilize this benchmark, dataset, or methodology in your research, please cite:

```bibtex
@inproceedings{danang_flood_benchmark_2022,
  title={Benchmarking Machine Learning Models for Road-Level Flood Prediction and Flood-Aware Routing},
  author={Pham, Thanh and Contributors},
  booktitle={IEEE Conference on Intelligent Transportation Systems},
  year={2026},
  address={Da Nang, Vietnam}
}
```
