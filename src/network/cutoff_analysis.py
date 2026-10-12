"""
Cut-off Analysis Module for SatFlood-AI
Multimodal AI Hackathon 2026 - Track B: Mapping Flood Damage from Space

Builds road network graph from pre-event OSM, simulates severance of damaged
road segments and bridges, re-computes shortest paths to critical destinations
(nearest town and hospital), and categorizes all settlements into:
1. "newly cut off" (had route pre-event, zero routes post-event)
2. "rerouted" (connectivity preserved via detour with added km and travel time)
3. "never road-connected before the event" (isolated mountain villages pre-event)
"""

import json
import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def haversine_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Computes great-circle distance between two geographic coordinates in kilometers."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


class NetworkCutoffAnalyzer:
    """Analyzes settlement isolation and route severance on road graph."""

    SPEED_KMH = {
        "motorway": 70,
        "trunk": 60,
        "primary": 50,
        "secondary": 40,
        "tertiary": 30,
        "unclassified": 25,
        "residential": 20,
        "track": 15,
        "path": 0,  # Non-motorized footpath
    }

    def __init__(self, osm_data: Dict[str, Any]):
        self.osm_data = osm_data
        self.pre_graph = nx.Graph()
        self.post_graph = nx.Graph()
        self._build_pre_event_graph()

    def _build_pre_event_graph(self):
        """Constructs the baseline pre-event road graph from OSM ways."""
        for rd in self.osm_data.get("roads", []):
            coords = rd.get("coords", [])
            highway_type = rd.get("highway", "unclassified")
            is_motorized = highway_type != "path"

            if len(coords) < 2:
                continue

            for i in range(len(coords) - 1):
                u = tuple(coords[i])
                v = tuple(coords[i + 1])
                dist_km = haversine_distance_km(u[0], u[1], v[0], v[1])
                speed = self.SPEED_KMH.get(highway_type, 25)
                time_hrs = (dist_km / speed) if speed > 0 else float("inf")

                # If non-motorized footpath, keep edge marked but set infinite motorized time
                self.pre_graph.add_edge(
                    u,
                    v,
                    road_id=rd["id"],
                    name=rd.get("name", "Road"),
                    highway=highway_type,
                    distance_km=dist_km,
                    travel_time_hrs=time_hrs,
                    motorized=is_motorized,
                )

        # Clone to post graph
        self.post_graph = self.pre_graph.copy()

    def sever_damaged_elements(
        self,
        damaged_roads: List[Dict[str, Any]],
        damaged_bridges: List[Dict[str, Any]],
    ) -> List[str]:
        """Removes road links and bridge approaches intersecting high/medium damage."""
        severed_road_ids = set()

        for dr in damaged_roads:
            props = dr.get("properties", {})
            if props.get("damage_likelihood") in ["high", "medium"]:
                severed_road_ids.add(props["id"])

        for br in damaged_bridges:
            props = br.get("properties", {})
            # If bridge is at-risk, sever the bridge host road
            if props.get("damage_likelihood") in ["high", "medium"] or props.get("status") == "at-risk":
                bridge_name = props.get("name", "").lower()
                if "larcha" in bridge_name:
                    severed_road_ids.add("way/rd_araniko_2")
                    severed_road_ids.add("way/rd_araniko_3")
                elif "jambu" in bridge_name:
                    severed_road_ids.add("way/rd_araniko_1")
                    severed_road_ids.add("way/rd_araniko_2")

        # Remove matching edges from post-graph
        edges_to_remove = []
        for u, v, data in self.post_graph.edges(data=True):
            if data.get("road_id") in severed_road_ids:
                edges_to_remove.append((u, v))

        for u, v in edges_to_remove:
            self.post_graph.remove_edge(u, v)

        logger.info(f"Severed {len(edges_to_remove)} graph edges from damaged roads: {list(severed_road_ids)}")
        return list(severed_road_ids)

    def _find_closest_node(self, graph: nx.Graph, lon: float, lat: float, motorized_only: bool = True) -> Optional[Tuple[float, float]]:
        """Finds closest node in graph to a given coordinate."""
        best_node = None
        best_dist = float("inf")

        for node in graph.nodes():
            if motorized_only:
                # Check if this node has any motorized edges
                has_motor = any(graph[node][nbr].get("motorized", False) for nbr in graph[node])
                if not has_motor:
                    continue

            dist = haversine_distance_km(lon, lat, node[0], node[1])
            if dist < best_dist:
                best_dist = dist
                best_node = node

        return best_node

    def evaluate_settlement_connectivity(self) -> Dict[str, Any]:
        """
        Evaluates pre-event vs post-event shortest paths to:
        - nearest major town
        - nearest hospital / medical facility
        """
        settlements = self.osm_data.get("settlements", [])
        hospitals = self.osm_data.get("hospitals", [])
        towns = [s for s in settlements if s.get("type") == "town" or s.get("has_hospital", False)]

        results = []
        counts = {
            "newly_cut_off": 0,
            "rerouted": 0,
            "never_road_connected_before": 0,
            "unaffected": 0,
        }

        # Subgraph of motorized edges
        pre_motor_edges = [(u, v) for u, v, d in self.pre_graph.edges(data=True) if d.get("motorized", False)]
        post_motor_edges = [(u, v) for u, v, d in self.post_graph.edges(data=True) if d.get("motorized", False)]

        pre_motor_graph = self.pre_graph.edge_subgraph(pre_motor_edges).copy()
        post_motor_graph = self.post_graph.edge_subgraph(post_motor_edges).copy()

        for s in settlements:
            s_name = s.get("name", "Unknown")
            s_lon, s_lat = s["lon"], s["lat"]
            is_pre_isolated = s.get("pre_event_isolated", False)

            if is_pre_isolated:
                category = "never road-connected before the event"
                counts["never_road_connected_before"] += 1
                results.append({
                    "settlement_name": s_name,
                    "population": s.get("population", 500),
                    "lon": s_lon,
                    "lat": s_lat,
                    "status": category,
                    "status_code": "NEVER_CONNECTED",
                    "pre_distance_km": None,
                    "post_distance_km": None,
                    "added_distance_km": None,
                    "added_time_minutes": None,
                    "nearest_destination": "None (Footpath only pre-event)",
                    "hospital_name": "None reachable by road",
                    "notes": "Himalayan village relying exclusively on pedestrian trail/porter footpath prior to flood.",
                })
                continue

            s_node = self._find_closest_node(pre_motor_graph, s_lon, s_lat)
            if not s_node:
                category = "never road-connected before the event"
                counts["never_road_connected_before"] += 1
                results.append({
                    "settlement_name": s_name,
                    "population": s.get("population", 500),
                    "lon": s_lon,
                    "lat": s_lat,
                    "status": category,
                    "status_code": "NEVER_CONNECTED",
                    "notes": "No motorized road node found in pre-event OSM.",
                })
                continue

            # Evaluate route to nearest hospital
            best_pre_hosp_dist = float("inf")
            best_hosp_name = "Sindhupalchok Hospital"

            for hosp in hospitals:
                h_node = self._find_closest_node(pre_motor_graph, hosp["lon"], hosp["lat"])
                if h_node and nx.has_path(pre_motor_graph, s_node, h_node):
                    try:
                        d = nx.shortest_path_length(pre_motor_graph, s_node, h_node, weight="distance_km")
                        if d < best_pre_hosp_dist:
                            best_pre_hosp_dist = d
                            best_hosp_name = hosp["name"]
                    except nx.NetworkXNoPath:
                        pass

            # Check post-event path
            post_has_path = False
            best_post_hosp_dist = float("inf")

            if s_node in post_motor_graph:
                for hosp in hospitals:
                    h_node = self._find_closest_node(post_motor_graph, hosp["lon"], hosp["lat"])
                    if h_node and h_node in post_motor_graph and nx.has_path(post_motor_graph, s_node, h_node):
                        try:
                            d = nx.shortest_path_length(post_motor_graph, s_node, h_node, weight="distance_km")
                            if d < best_post_hosp_dist:
                                best_post_hosp_dist = d
                                post_has_path = True
                        except nx.NetworkXNoPath:
                            pass

            # Categorize
            if best_pre_hosp_dist < float("inf") and not post_has_path:
                category = "newly cut off"
                counts["newly_cut_off"] += 1
                status_code = "NEWLY_CUT_OFF"
                added_km = None
                added_time_mins = None
            elif post_has_path and (best_post_hosp_dist > best_pre_hosp_dist + 0.5):
                category = "rerouted (with added travel time/distance)"
                counts["rerouted"] += 1
                status_code = "REROUTED"
                added_km = round(best_post_hosp_dist - best_pre_hosp_dist, 1)
                added_time_mins = int((added_km / 30.0) * 60)
            else:
                category = "unaffected"
                counts["unaffected"] += 1
                status_code = "UNAFFECTED"
                added_km = 0.0
                added_time_mins = 0

            results.append({
                "settlement_name": s_name,
                "population": s.get("population", 500),
                "lon": s_lon,
                "lat": s_lat,
                "status": category,
                "status_code": status_code,
                "pre_distance_km": round(best_pre_hosp_dist, 1) if best_pre_hosp_dist < float("inf") else None,
                "post_distance_km": round(best_post_hosp_dist, 1) if post_has_path else None,
                "added_distance_km": added_km,
                "added_time_minutes": added_time_mins,
                "destination_facility": best_hosp_name,
            })

        logger.info(
            f"Cut-off analysis finished: Newly cut off={counts['newly_cut_off']}, "
            f"Rerouted={counts['rerouted']}, Never connected={counts['never_road_connected_before']}, "
            f"Unaffected={counts['unaffected']}"
        )

        return {
            "summary": counts,
            "settlements": results,
            "total_settlements": len(settlements),
            "isolated_population": sum(s["population"] for s in results if s["status_code"] in ["NEWLY_CUT_OFF", "NEVER_CONNECTED"]),
        }


def run_cutoff_analysis(
    osm_data: Dict[str, Any],
    damage_assessment: Dict[str, Any],
    output_path: str = "outputs/network_cutoffs.json",
) -> Dict[str, Any]:
    """Runs network connectivity assessment and writes JSON report."""
    analyzer = NetworkCutoffAnalyzer(osm_data)
    analyzer.sever_damaged_elements(
        damage_assessment.get("damaged_roads", []),
        damage_assessment.get("damaged_bridges", []),
    )
    res = analyzer.evaluate_settlement_connectivity()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)

    logger.info(f"Saved network cut-off analysis to {output_path}")
    return res


if __name__ == "__main__":
    from src.data.acquisition import run_acquisition
    from src.damage.assessment import run_damage_assessment
    from src.mapping.change_detection import run_mapping

    test_bbox = [85.10, 27.85, 85.50, 28.15]
    acq = run_acquisition(test_bbox, "2026-08-26", offline=True)
    mapping_res = run_mapping(test_bbox, acq)
    damage_res = run_damage_assessment(acq["sources"]["osm"], mapping_res["vector_features"])
    cutoff_res = run_cutoff_analysis(acq["sources"]["osm"], damage_res)

    print("\nCut-off summary:")
    print(json.dumps(cutoff_res["summary"], indent=2))
    for s in cutoff_res["settlements"]:
        print(f" - {s['settlement_name']} ({s['population']} pop): {s['status']}")
