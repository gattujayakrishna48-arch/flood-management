"""Unit tests for Damage Assessment and Spatial Intersection with Buffers."""

import pytest
from src.damage.assessment import DamageAssessor


def test_building_damage_intersection():
    """Verifies that building footprint intersected by hazard polygon is flagged."""
    assessor = DamageAssessor(building_buffer_deg=0.001)

    osm_data = {
        "buildings": [
            {
                "id": "bld_1",
                "settlement": "TestVillage",
                "footprint": [[85.40, 27.90], [85.41, 27.90], [85.41, 27.91], [85.40, 27.91], [85.40, 27.90]],
            },
            {
                "id": "bld_2_safe",
                "settlement": "SafeVillage",
                "footprint": [[85.10, 28.10], [85.11, 28.10], [85.11, 28.11], [85.10, 28.11], [85.10, 28.10]],
            },
        ],
        "roads": [],
        "bridges": [],
    }

    hazards = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[85.39, 27.89], [85.42, 27.89], [85.42, 27.92], [85.39, 27.92], [85.39, 27.89]]],
            },
            "properties": {"class": "flood_water", "confidence": 0.85},
        }
    ]

    res = assessor.assess(osm_data, hazards)
    assert res["summary"]["damaged_buildings_count"] == 1
    assert res["damaged_buildings"][0]["properties"]["id"] == "bld_1"


def test_bridge_at_risk_detection():
    """Verifies bridge approach hazard triggers at-risk status."""
    assessor = DamageAssessor(bridge_buffer_deg=0.002)

    osm_data = {
        "buildings": [],
        "roads": [],
        "bridges": [
            {"id": "bridge_1", "name": "Canyon Bridge", "lon": 85.45, "lat": 27.90, "highway": "primary", "length_m": 50}
        ],
    }

    hazards = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[85.449, 27.899], [85.452, 27.899], [85.452, 27.902], [85.449, 27.902], [85.449, 27.899]]],
            },
            "properties": {"class": "flood_water", "confidence": 0.90},
        }
    ]

    res = assessor.assess(osm_data, hazards)
    assert res["summary"]["bridges_at_risk_count"] == 1
    assert res["damaged_bridges"][0]["properties"]["status"] == "at-risk"
