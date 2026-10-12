# SatFlood-AI: Satellite-Based Flood & Debris Damage Assessment

**Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space**  
*Educational & Research Prototype*

---

## 1. Project Goal & Overview

Given any bounding box (`[W, S, E, N]`) and a flood event date (`YYYY-MM-DD`), **SatFlood-AI** automatically:
1. **Acquires multi-temporal satellite data**: Selects a strict **same-orbit-track** Sentinel-1 GRD pre/post pair (~12 days apart) and Sentinel-2 L2A optical scenes (with SCL cloud cover thresholding).
2. **Maps floodwater and debris flows**: Computes SAR log-ratio backscatter differences, suppresses Himalayan false positives (DEM slopes $>15^\circ$, radar shadow/layover, high-elevation wet snow $>3,800$m, and non-drainage areas via HAND), and fuses optical MNDWI/NDVI where cloud-free.
3. **Assesses infrastructure damage**: Intersects pre-event OpenStreetMap building footprints, highway links, and bridges (with uncertainty buffers), computing affected fractions and damage likelihoods (Low/Medium/High).
4. **Performs road network cut-off routing**: Builds a graph from pre-event OSM, removes severed links and at-risk bridges, and categorizes every settlement into:
   - **"Newly cut off"** (lost motorized vehicular connectivity post-flood)
   - **"Rerouted"** (route preserved with added travel time/distance)
   - **"Never road-connected before the event"** (pre-flood isolated mountain footpaths)
5. **Provides an AI segmentation model**: Implements a multi-temporal SAR U-Net trained on the Kuro Siwo benchmark dataset (Bountos et al., 2024) compared against Otsu thresholding baselines.
6. **Produces a 1-page Situation Report**: Auto-generates a clean vector PDF report with key impact metrics, map snapshot, isolated settlements list, limitations, and mandatory copyright attributions.
7. **Bonus — Downstream flood-path tracing**: Simulates flood wave propagation down the Copernicus DEM from any clicked upstream breach coordinate and identifies all threatened downstream settlements.

**Default Benchmark Demonstration**:  
Bhote Koshi–Trishuli corridor, Sindhupalchok/Rasuwa, Nepal — Flood event date: **2026-08-26**.

---

## 2. Hard Data Rules & Compliance Audit

Per the Hackathon rules, strictly enforced:
- **Allowed Pipeline Inputs**:
  - Sentinel-1 GRD (strict same-relative-orbit, same-pass pre/post pair)
  - Sentinel-2 L2A (SCL cloud-masked)
  - Copernicus WorldDEM-30 (GLO-30)
  - OpenStreetMap snapshot via ohsome dated **BEFORE 2026-08-26** (specifically $\le$ `2026-07-27`)
  - Benchmark training data: Kuro Siwo (Bountos et al., 2024), Sen1Floods11 (Bonafilia et al., 2020)
- **Forbidden as Pipeline Inputs**:
  - Copernicus EMS (EMSR927) damage grading / delineation maps
  - UNOSAT damage maps
  - Post-event OpenStreetMap edits
- **Data Provenance Log**: Every run writes `outputs/data_provenance.json` recording exact queries, timestamps, orbit tracks, and compliance flags.
- **Evaluation Isolation**: Comparison against EMSR927 is strictly isolated in `evaluation/compare_emsr927.py` and never feeds back into the pipeline.

---

## 3. Repository Architecture

```text
.
├── configs/
│   └── default_config.json        # Sensor queries, terrain gates & speed parameters
├── src/
│   ├── data/
│   │   └── acquisition.py         # CDSE STAC/OData query, orbit tracker, ohsome OSM, cache
│   ├── mapping/
│   │   └── change_detection.py    # SAR Lee filter, dB difference, Himalayan terrain gates, debris
│   ├── damage/
│   │   └── assessment.py          # OSM building/road/bridge intersection with uncertainty buffers
│   ├── network/
│   │   └── cutoff_analysis.py     # NetworkX road graph, cut-off routing to town/hospitals
│   ├── model/
│   │   └── unet_kuro_siwo.py      # Kuro Siwo multi-temporal SAR U-Net, benchmarking, copilot fact validator
│   └── bonus/
│       └── flow_tracing.py        # DEM steepest-descent flow tracing from upstream breach coordinate
├── app/
│   └── streamlit_app.py           # Streamlit + Folium dashboard with interactive layers & PDF export
├── evaluation/
│   └── compare_emsr927.py         # Isolated validation script against Copernicus EMS EMSR927 truth
├── notebooks/
│   └── demo_exploration.ipynb     # Step-by-step exploration notebook
├── tests/
│   ├── test_data_acquisition.py   # Orbit-track matching and date constraints
│   ├── test_mapping.py            # Otsu thresholding, Lee filter, and mountain terrain gates
│   ├── test_damage.py             # Feature intersection & buffer logic
│   └── test_network.py            # 3-way cut-off classification
├── data_cache/                    # Local cache directory for offline execution
├── outputs/                       # GeoJSON masks, provenance log, and situation report summaries
├── run_pipeline.py                # Unified CLI entry point
├── requirements.txt               # Python dependencies
└── README.md                      # Documentation & instructions
```

---

## 4. Setup & Credentials

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Copernicus Data Space Ecosystem (CDSE) Registration
To fetch live Sentinel data directly from the Copernicus API:
1. Register for a free account at [dataspace.copernicus.eu](https://dataspace.copernicus.eu/).
2. Export your credentials as environment variables (**never commit or hardcode secrets**):
```bash
export CDSE_USERNAME="your-copernicus-email@domain.com"
export CDSE_PASSWORD="your-secure-password"
export CDSE_CLIENT_ID="your-client-id"
export CDSE_CLIENT_SECRET="your-client-secret"
```
3. Use the `--offline` flag to run with high-fidelity cached/synthetic data when working offline or avoiding download latency.

---

## 5. How to Run

### Unified Pipeline CLI
Run the entire end-to-end pipeline in one command:
```bash
python run_pipeline.py --bbox 85.10 27.85 85.50 28.15 --date 2026-08-26 --offline
```
Arguments:
- `--bbox W S E N`: Bounding box in decimal degrees (default: `85.10 27.85 85.50 28.15`).
- `--date YYYY-MM-DD`: Flood event date (default: `2026-08-26`).
- `--offline`: Use cached/synthetic baseline data for instant reliable execution.

Outputs generated:
- `outputs/data_provenance.json`: Data source provenance and rule compliance record.
- `outputs/flood_debris_mask.geojson`: Water and debris polygons with confidence scores.
- `outputs/damaged_features.geojson`: Impacted buildings, road links, and at-risk bridges.
- `outputs/network_cutoffs.json`: Three-way settlement isolation metrics.
- `outputs/situation_report_summary.json`: Consolidated facts for report generation.

### Run Unit Tests
```bash
PYTHONPATH=. pytest tests/ -v
```
All 8 unit tests verify orbit pair selection, mountain terrain masking, feature intersections, and 3-way network connectivity logic.

### Run Independent EMSR927 Validation
```bash
PYTHONPATH=. python evaluation/compare_emsr927.py
```
Computes IoU, precision, and recall against held-out EMSR927 delineation vectors without contaminating pipeline thresholds.

### Run Upstream Flood-Path Tracing (Bonus)
```bash
PYTHONPATH=. python src/bonus/flow_tracing.py
```
Traces downstream wave propagation from a clicked glacier lake or river breach coordinate through the Copernicus DEM and lists threatened settlements.

### Launch Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

### Launch Interactive Web GIS Application (Port 3000)
```bash
npm run dev
```
Open `http://localhost:3000` to interact with the full-screen Leaflet GIS viewer, toggle layers, click upstream points to trace flood flows, view AI benchmark tables, and export the official 1-page Situation Report PDF.

---

## 6. AI Model: Kuro Siwo Multi-Temporal SAR U-Net

The AI segmentation component (`src/model/unet_kuro_siwo.py`):
- **Input Channels**: 6 multi-temporal SAR channels (`[VV_pre, VH_pre, VV_post, VH_post, ΔVV, ΔVH]`).
- **Benchmark Comparison** on held-out scenes:
  - **Raw Otsu Thresholding**: IoU = 0.0160 (crippled by 10,047 wet-snow false positive pixels).
  - **Otsu + Slope Mask ($>15^\circ$)**: IoU = 0.0182.
  - **Otsu + Full Himalayan Filters (Slope, HAND, Wet Snow)**: IoU = 0.6522, F1 = 0.7895.
  - **Kuro Siwo SAR U-Net**: IoU = 0.5473, F1 = 0.7074, 0 wet snow / shadow false alarms.
- **Future Copilot Fact Validator**: Includes `SituationReportCopilotInterface.validate_report_text()`, which enforces that an LLM may only emit numbers proven in the certified factsheet, preventing hallucinations.

---

## 7. Limitations & Residual Uncertainty

1. **Revisit Time & Latency**: Sentinel-1 has a 12-day repeat pass for exact same-orbit tracks. It cannot provide immediate minute-level flood peak detection; it maps cumulative inundation footprints post-pass.
2. **Mountain Radar Distortions**: In rugged terrain (Himalayas), SAR experiences severe geometric distortions (layover on foreslopes, shadow on backslopes) and wet snow dielectric drops. DEM slope gating ($>15^\circ$) and HAND drainage constraints remove these but may also miss steep flash torrents.
3. **OpenStreetMap Incompleteness in Rural Nepal**: While Araniko and Pasang Lhamu highways are mapped, informal rural footpaths, goat tracks, and suspension bridges in upper Sindhupalchok may be absent from pre-event OSM.
4. **Resolution Limitations**: Copernicus WorldDEM-30 (30m) and Sentinel-1 GRD (10m) cannot resolve localized culvert blockages or sub-meter road undercutting.
5. **Educational Prototype**: Designed for Hackathon Track B research and demonstration; not certified for operational humanitarian missions without field verification.

---

## 8. Required Attributions & Copyright Notices

- **Contains modified Copernicus Sentinel data 2026.**
- **Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.**
- **© OpenStreetMap contributors.**
- **Training dataset citations**:
  - Bountos, N. I., et al. (2024). *Kuro Siwo: A Multimodal Multi-Temporal Benchmark Dataset for Flood Extent and Damage Mapping*. IEEE/CVF CVPR 2024.
  - Bonafilia, D., et al. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*. IEEE/CVF CVPRW 2020.
