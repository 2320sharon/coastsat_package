"""End-to-end test of the SAR read path, from a session on disk to the image thresholded.

Walks get_filepath -> get_filenames -> preprocess_image -> get_sar_otsu_image against
real .tif files. A dual-polarization scene must reach Otsu as the mean of VV and VH,
while a scene with only VH must still threshold VH exactly as it did before.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat import SDS_tools
from coastsat.SDS_preprocess import preprocess_image
from coastsat.SDS_shoreline import get_sar_band_index, get_sar_otsu_image

DATE = "2023-12-03-19-15-49"
SITENAME = "site"
SIZE = 8

# a distinct constant per polarization, so the channel that ends up being read is obvious
VALUES = {"VV": -12.0, "VH": -22.0, "HH": -32.0}


def write_tif(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), SIZE, SIZE, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform([500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    dataset.GetRasterBand(1).WriteArray(np.full((SIZE, SIZE), value, dtype="float64"))
    dataset = None


def make_session(root, polarizations):
    for polar in polarizations:
        write_tif(
            root / SITENAME / "S1" / polar / f"{DATE}_S1_{SITENAME}_{polar}.tif",
            VALUES[polar],
        )
    return {"filepath": str(root), "sitename": SITENAME}


def read_scene(root, polarizations, primary=None):
    """Resolve and read a scene exactly the way extract_shorelines does."""
    inputs = make_session(root, polarizations)
    filepath = SDS_tools.get_filepath(inputs, "S1")
    filename = f"{DATE}_S1_{SITENAME}_{primary or polarizations[0]}.tif"
    fn = SDS_tools.get_filenames(filename, filepath, "S1")
    im_ms, georef, cloud_mask, im_nodata = preprocess_image(fn, "S1", {}, "C02")
    return fn, im_ms, georef, cloud_mask, im_nodata


def test_dual_polarization_scene_returns_the_composite(tmp_path):
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VV", "VH"])

    assert len(fn) == 2
    assert im_ms.shape == (SIZE, SIZE, 3)  # [VV, VH, VV-VH]
    assert np.all(im_ms[:, :, 0] == VALUES["VV"])
    assert np.all(im_ms[:, :, 1] == VALUES["VH"])
    assert np.all(im_ms[:, :, 2] == VALUES["VV"] - VALUES["VH"])


def test_dual_polarization_scene_thresholds_the_mean_of_vv_and_vh(tmp_path):
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VV", "VH"])

    im_sar = get_sar_otsu_image(im_ms, fn)

    assert np.all(im_sar == (VALUES["VV"] + VALUES["VH"]) / 2.0)


def test_legacy_vh_only_session_is_unchanged(tmp_path):
    # no VV, so no composite: it stays a single band and Otsu still sees plain VH
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VH"])

    assert im_ms.shape == (SIZE, SIZE, 1)
    assert np.all(get_sar_otsu_image(im_ms, fn) == VALUES["VH"])


def test_mixed_session_thresholds_vh_for_a_legacy_scene(tmp_path):
    # a VV+VH session that also contains a scene downloaded before VV was added
    inputs = make_session(tmp_path, ["VV", "VH"])
    (tmp_path / SITENAME / "S1" / "VV" / f"{DATE}_S1_{SITENAME}_VV.tif").unlink()
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(f"{DATE}_S1_{SITENAME}_VH.tif", filepath, "S1")
    im_ms, _, _, _ = preprocess_image(fn, "S1", {}, "C02")

    assert im_ms.shape == (SIZE, SIZE, 1)
    assert np.all(get_sar_otsu_image(im_ms, fn) == VALUES["VH"])


def test_vv_only_session_thresholds_vv(tmp_path):
    fn, im_ms, _, _, _ = read_scene(tmp_path, ["VV"])

    assert im_ms.shape == (SIZE, SIZE, 1)
    assert np.all(get_sar_otsu_image(im_ms, fn) == VALUES["VV"])
    assert get_sar_band_index(fn) == 0


def test_masks_match_the_image_footprint(tmp_path):
    _, im_ms, _, cloud_mask, im_nodata = read_scene(tmp_path, ["VV", "VH"])

    # SAR carries no cloud or no-data information, but the masks must still be 2D
    assert cloud_mask.shape == im_ms.shape[:2]
    assert im_nodata.shape == im_ms.shape[:2]
    assert not cloud_mask.any()
    assert not im_nodata.any()


def test_channel_order_does_not_depend_on_the_metadata_filename(tmp_path):
    # the metadata records the VV file, but starting from VH must give the same stack
    _, from_vv, _, _, _ = read_scene(tmp_path, ["VV", "VH"], primary="VV")
    _, from_vh, _, _, _ = read_scene(tmp_path, ["VV", "VH"], primary="VH")

    assert np.array_equal(from_vv, from_vh)
