"""Unit tests for Flood & Debris Mapping and False Positive Suppression."""

import numpy as np
import pytest
from src.mapping.change_detection import otsu_threshold, sar_lee_filter_db, FloodDebrisMapper


def test_otsu_threshold_bimodal():
    """Verifies Otsu threshold cleanly separates synthetic bimodal drop."""
    np.random.seed(42)
    # Background (dry land, small delta around 0 dB)
    dry = np.random.normal(0.0, 0.5, 500)
    # Foreground (water, large negative drop around -6 dB)
    water = np.random.normal(-6.0, 0.5, 500)
    combined = np.concatenate([dry, water])

    t = otsu_threshold(combined)
    assert -5.5 < t < -1.5


def test_sar_lee_filter_noise_reduction():
    """Verifies Lee filter smooths speckle noise while preserving mean."""
    np.random.seed(42)
    clean = np.full((30, 30), -15.0)
    noisy = clean + np.random.normal(0, 3.0, (30, 30))

    filtered = sar_lee_filter_db(noisy, size=5)
    assert np.var(filtered) < np.var(noisy)


def test_himalayan_mountain_masking():
    """Verifies that steep slopes (>15 deg) and snow (>3800m) are masked out."""
    mapper = FloodDebrisMapper(grid_size=(10, 10))
    scene = {
        "s1_pre_vv": np.full((10, 10), -12.0),
        "s1_post_vv": np.full((10, 10), -22.0),  # Strong negative drop
        "s1_pre_vh": np.full((10, 10), -18.0),
        "s1_post_vh": np.full((10, 10), -28.0),
        "slope_deg": np.full((10, 10), 25.0),    # Steep slope > 15 deg
        "dem": np.full((10, 10), 4000.0),         # High elevation wet snow
        "hand": np.full((10, 10), 30.0),          # Far from drainage
        "shadow_zone": np.zeros((10, 10), dtype=bool),
        "river_mask": np.zeros((10, 10), dtype=bool),
        "s2_ndvi_pre": np.full((10, 10), 0.6),
        "s2_ndvi_post": np.full((10, 10), 0.6),
        "s2_mndwi": np.full((10, 10), -0.3),
        "lons": np.linspace(85.1, 85.5, 10),
        "lats": np.linspace(28.1, 27.8, 10),
    }

    res = mapper.process(scene, use_optical=False)
    # All pixels should be rejected due to steep slope, high DEM, and high HAND
    assert np.sum(res["class_grid"] == 1) == 0
