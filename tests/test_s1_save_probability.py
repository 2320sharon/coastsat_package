"""Tests that P(water) of a model-segmented S1 scene is written as a GeoTIFF.

segment_water always returned the probability but the detection branch used to discard
it. save_sar_probability writes it to S1/prob/, on the grid of the scene's first
polarization file, with invalid pixels as NaN. Losing the raster must never cost the
scene its shoreline, so failures are logged and swallowed.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal, osr

from coastsat.SDS_shoreline import compute_sar_water_fraction, save_sar_probability

DATE = "2023-12-03-19-15-49"
SIZE = 8
GEOTRANSFORM = (500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0)


def write_scene_tif(path, size=SIZE):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32756)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), -20.0))
    dataset = None
    return str(path)


@pytest.fixture
def scene(tmp_path):
    """A dual-polarization scene layout: S1/VV and S1/VH each holding one .tif."""
    fn = [
        write_scene_tif(tmp_path / "S1" / polar / f"{DATE}_S1_site_{polar}.tif")
        for polar in ("VV", "VH")
    ]
    return fn


def probability():
    prob = np.linspace(0.0, 1.0, SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    return prob


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(str(message))


# ------------------------------------------------------------------ the happy path


def test_writes_the_probability_next_to_the_polarization_folders(scene, tmp_path):
    valid = np.ones((SIZE, SIZE), dtype=bool)

    path = save_sar_probability(probability(), valid, scene)

    assert path is not None
    assert Path(path) == tmp_path / "S1" / "prob" / f"{DATE}_S1_site_prob.tif"
    assert Path(path).exists()


def test_values_and_dtype_survive_the_round_trip(scene):
    valid = np.ones((SIZE, SIZE), dtype=bool)
    prob = probability()

    path = save_sar_probability(prob, valid, scene)

    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1).ReadAsArray()
    assert band.dtype == np.float32
    np.testing.assert_allclose(band, prob, rtol=1e-6)


def test_invalid_pixels_are_nan_and_nodata_is_declared(scene):
    valid = np.ones((SIZE, SIZE), dtype=bool)
    valid[:, :2] = False

    path = save_sar_probability(probability(), valid, scene)

    dataset = gdal.Open(path)
    band = dataset.GetRasterBand(1)
    data = band.ReadAsArray()
    assert np.all(np.isnan(data[:, :2]))
    assert np.all(np.isfinite(data[:, 2:]))
    # NaN, not 0.0: a probability can genuinely be 0.0
    assert np.isnan(band.GetNoDataValue())


def test_the_grid_is_copied_from_the_scene(scene):
    valid = np.ones((SIZE, SIZE), dtype=bool)

    path = save_sar_probability(probability(), valid, scene)

    dataset = gdal.Open(path)
    assert dataset.GetGeoTransform() == GEOTRANSFORM
    assert "32756" in dataset.GetProjection()


def test_the_dup_suffix_is_kept(tmp_path):
    # both polarizations of a duplicate scene share a _dupN token, which is what
    # pairs them; the probability file has to carry it too
    fn = [
        write_scene_tif(tmp_path / "S1" / polar / f"{DATE}_S1_site_{polar}_dup0.tif")
        for polar in ("VV", "VH")
    ]

    path = save_sar_probability(probability(), np.ones((SIZE, SIZE), dtype=bool), fn)

    assert Path(path).name == f"{DATE}_S1_site_prob_dup0.tif"


def test_accepts_a_bare_string_path(tmp_path):
    fn = write_scene_tif(tmp_path / "S1" / "VH" / f"{DATE}_S1_site_VH.tif")

    path = save_sar_probability(probability(), np.ones((SIZE, SIZE), dtype=bool), fn)

    assert Path(path).exists()


# ------------------------------------------------------------------ failure handling


def test_an_unreadable_scene_is_logged_and_swallowed(tmp_path):
    logger = RecordingLogger()
    missing = str(tmp_path / "S1" / "VV" / "gone.tif")

    path = save_sar_probability(
        probability(),
        np.ones((SIZE, SIZE), dtype=bool),
        [missing],
        logger=logger,
        date_str=DATE,
    )

    assert path is None
    assert any(DATE in warning for warning in logger.warnings)


# ------------------------------------------------------------- compute_sar_water_fraction


def test_water_fraction_is_computed_over_the_region():
    water = np.zeros((4, 4), dtype=bool)
    water[2:, :] = True
    region = np.ones((4, 4), dtype=bool)

    assert compute_sar_water_fraction(water, region) == pytest.approx(0.5)


def test_water_fraction_ignores_pixels_outside_the_region():
    water = np.zeros((4, 4), dtype=bool)
    water[2:, :] = True
    region = np.zeros((4, 4), dtype=bool)
    region[2:, :] = True  # only water pixels are in the region

    assert compute_sar_water_fraction(water, region) == pytest.approx(1.0)


def test_water_fraction_of_an_empty_region_is_nan():
    # an empty buffer & valid intersection must not divide by zero
    water = np.zeros((4, 4), dtype=bool)
    region = np.zeros((4, 4), dtype=bool)

    assert np.isnan(compute_sar_water_fraction(water, region))
