"""Unit tests for Network Connectivity and 3-way Cut-off Categorization."""

import pytest
from src.network.cutoff_analysis import NetworkCutoffAnalyzer


def test_three_way_cutoff_categorization():
    """
    Tests the 3 distinct settlement categories:
    1. 'newly cut off'
    2. 'rerouted' (with added distance)
    3. 'never road-connected before the event'
    """
    osm_data = {
        "settlements": [
            {"name": "TownA", "lon": 85.0, "lat": 27.0, "type": "town", "population": 10000},
            {"name": "VillageCutOff", "lon": 85.2, "lat": 27.0, "type": "village", "population": 2000},
            {"name": "VillageRerouted", "lon": 85.1, "lat": 27.2, "type": "village", "population": 1500},
            {"name": "RemoteHamlet", "lon": 85.3, "lat": 27.3, "type": "hamlet", "population": 300, "pre_event_isolated": True},
        ],
        "hospitals": [
            {"name": "District Hospital", "lon": 85.0, "lat": 27.0, "beds": 40}
        ],
        "roads": [
            # Main highway TownA <-> VillageCutOff
            {"id": "rd_main", "highway": "primary", "coords": [[85.0, 27.0], [85.1, 27.0], [85.2, 27.0]], "length_km": 20},
            # Road to VillageRerouted direct
            {"id": "rd_direct_reroute", "highway": "secondary", "coords": [[85.0, 27.0], [85.1, 27.2]], "length_km": 15},
            # Detour road to VillageRerouted
            {"id": "rd_detour_reroute", "highway": "track", "coords": [[85.0, 27.0], [85.0, 27.3], [85.1, 27.2]], "length_km": 35},
        ],
        "bridges": [],
    }

    analyzer = NetworkCutoffAnalyzer(osm_data)

    # Damage: sever rd_main and rd_direct_reroute
    damaged_roads = [
        {"properties": {"id": "rd_main", "damage_likelihood": "high"}},
        {"properties": {"id": "rd_direct_reroute", "damage_likelihood": "high"}},
    ]
    analyzer.sever_damaged_elements(damaged_roads, [])
    eval_res = analyzer.evaluate_settlement_connectivity()

    status_by_name = {s["settlement_name"]: s["status_code"] for s in eval_res["settlements"]}

    assert status_by_name["VillageCutOff"] == "NEWLY_CUT_OFF"
    assert status_by_name["VillageRerouted"] == "REROUTED"
    assert status_by_name["RemoteHamlet"] == "NEVER_CONNECTED"
    assert status_by_name["TownA"] == "UNAFFECTED"
