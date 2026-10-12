#!/usr/bin/env python3
"""
SatFlood-AI: Main End-to-End Pipeline CLI
Multimodal AI Hackathon 2026, Track B: Mapping Flood Damage from Space

Usage:
  python run_pipeline.py --bbox W S E N --date YYYY-MM-DD [--offline]

Default demo:
  python run_pipeline.py --bbox 85.10 27.85 85.50 28.15 --date 2026-08-26 --offline
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import List

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.acquisition import run_acquisition
from src.mapping.change_detection import run_mapping
from src.damage.assessment import run_damage_assessment
from src.network.cutoff_analysis import run_cutoff_analysis
from src.model.unet_kuro_siwo import SituationReportCopilotInterface

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SatFlood-Pipeline")


def run_pipeline(
    bbox: List[float],
    event_date: str,
    offline: bool = True,
    output_dir: str = "outputs",
):
    """Executes the complete satellite flood damage assessment pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=" * 70)
    logger.info("SatFlood-AI: Satellite Flood & Debris Damage Assessment Pipeline")
    logger.info(f"Target Region BBox: {bbox}")
    logger.info(f"Flood Event Date  : {event_date}")
    logger.info(f"Offline Mode Flag : {offline}")
    logger.info("=" * 70)

    # 1. Data Acquisition (Sentinel-1, Sentinel-2, DEM, pre-event OSM)
    logger.info("[1/4] Running Data Acquisition & Provenance Verification...")
    acq_res = run_acquisition(bbox, event_date, offline=offline)

    # 2. Flood and Debris Mapping (SAR Speckle Filtering, dB difference, Himalayan terrain masking)
    logger.info("[2/4] Running Flood and Debris Mapping (SAR + Optical + DEM)...")
    mask_path = os.path.join(output_dir, "flood_debris_mask.geojson")
    mapping_res = run_mapping(bbox, acq_res, output_path=mask_path)

    # 3. Infrastructure Damage Assessment (Buildings, Roads, Bridges)
    logger.info("[3/4] Running Infrastructure Damage Assessment (OSM Overlays)...")
    damage_path = os.path.join(output_dir, "damaged_features.geojson")
    damage_res = run_damage_assessment(
        acq_res["sources"]["osm"],
        mapping_res["vector_features"],
        output_path=damage_path,
    )

    # 4. Road Network Connectivity & Cut-off Analysis
    logger.info("[4/4] Running Road Network Connectivity & Cut-Off Routing...")
    cutoff_path = os.path.join(output_dir, "network_cutoffs.json")
    cutoff_res = run_cutoff_analysis(
        acq_res["sources"]["osm"],
        damage_res,
        output_path=cutoff_path,
    )

    # Certified Facts Extraction (for clean copilot and situation report generation)
    facts = SituationReportCopilotInterface.extract_certified_facts(
        acq_res, mapping_res, damage_res, cutoff_res
    )

    summary_payload = {
        "execution_date": datetime.utcnow().isoformat() + "Z",
        "bbox": bbox,
        "flood_date": event_date,
        "offline_execution": offline,
        "certified_facts": facts,
        "mapping_summary": mapping_res["summary"],
        "damage_summary": damage_res["summary"],
        "cutoff_summary": cutoff_res["summary"],
        "settlements": cutoff_res["settlements"],
        "attributions": [
            "Contains modified Copernicus Sentinel data 2026.",
            "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
            "© OpenStreetMap contributors.",
            "Training dataset citations: Bountos et al., 2024 (Kuro Siwo); Bonafilia et al., 2020 (Sen1Floods11).",
        ],
    }

    summary_file = os.path.join(output_dir, "situation_report_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info(f"Situation report summary saved to {summary_file}")
    logger.info("=" * 70)
    logger.info("PIPELINE EXECUTION COMPLETE - ALL OUTPUTS READY")
    logger.info(f" * Flooded Area       : {facts['flooded_area_km2']} km²")
    logger.info(f" * Debris Area        : {facts['debris_area_km2']} km²")
    logger.info(f" * Damaged Buildings  : {facts['damaged_buildings_count']}")
    logger.info(f" * Blocked Highways   : {facts['blocked_roads_km']} km")
    logger.info(f" * Bridges at Risk    : {facts['bridges_at_risk_count']}")
    logger.info(f" * Newly Cut Off      : {facts['newly_cut_off_settlements']} settlements")
    logger.info(f" * Isolated Population: {facts['isolated_population']} people")
    logger.info("=" * 70)

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SatFlood-AI: End-to-End Satellite Flood & Debris Damage Assessment Pipeline"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[85.10, 27.85, 85.50, 28.15],
        metavar=("W", "S", "E", "N"),
        help="Bounding box in decimal degrees (e.g. 85.10 27.85 85.50 28.15)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2026-08-26",
        help="Flood event date in YYYY-MM-DD format (default: 2026-08-26)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=True,
        help="Use cached/synthetic fallback data to guarantee reliable execution without network latency",
    )

    args = parser.parse_args()
    run_pipeline(bbox=args.bbox, event_date=args.date, offline=args.offline)
