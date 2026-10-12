"""Unit tests for Orbit Pair Selection and Acquisition rules."""

import pytest
from src.data.acquisition import OrbitPairSelector


def test_same_orbit_selection_success():
    """Verifies that selector picks scenes with the exact same relative orbit and direction."""
    acquisitions = [
        {"id": "A1", "date": "2026-08-14T10:00:00Z", "relative_orbit": 121, "orbit_direction": "ASCENDING"},
        {"id": "B1", "date": "2026-08-18T10:00:00Z", "relative_orbit": 19, "orbit_direction": "DESCENDING"},
        {"id": "A2", "date": "2026-08-26T10:00:00Z", "relative_orbit": 121, "orbit_direction": "ASCENDING"},
        {"id": "B2", "date": "2026-08-30T10:00:00Z", "relative_orbit": 19, "orbit_direction": "DESCENDING"},
    ]
    pre, post = OrbitPairSelector.select_best_pair(acquisitions, "2026-08-26", target_interval_days=12)

    assert pre["id"] == "A1"
    assert post["id"] == "A2"
    assert pre["relative_orbit"] == post["relative_orbit"] == 121
    assert pre["orbit_direction"] == post["orbit_direction"] == "ASCENDING"


def test_mismatched_orbits_rejected():
    """Verifies that pairs with different relative orbits are never mixed (HARD RULE)."""
    acquisitions = [
        {"id": "A1", "date": "2026-08-14T10:00:00Z", "relative_orbit": 121, "orbit_direction": "ASCENDING"},
        # Only orbit 19 is available post-event
        {"id": "B2", "date": "2026-08-26T10:00:00Z", "relative_orbit": 19, "orbit_direction": "DESCENDING"},
    ]
    with pytest.raises(ValueError, match="No matching same-orbit-track pair found"):
        OrbitPairSelector.select_best_pair(acquisitions, "2026-08-26")
