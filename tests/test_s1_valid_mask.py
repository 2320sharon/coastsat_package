"""Tests for the per-pixel validity mask of a SAR scene.

The segmentation model neutralises invalid pixels on input and blanks them out of the
prediction, so this mask decides which pixels are allowed to become a land/water call.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat.SDS_tools import read_sar_valid_mask

GEOTRANSFORM = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0]
SIZE = 8


def write_tif(path, array, nodata=None):
    """Write a single-band Float64 .tif, like an Earth Engine S1 download."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), array.shape[1], array.shape[0], 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    band = dataset.GetRasterBand(1)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    band.WriteArray(array.astype("float64"))
    dataset = None
    return str(path)


def filled(value):
    return np.full((SIZE, SIZE), value, dtype="float64")


def test_all_valid_when_no_nodata_is_declared(tmp_path):
    vv = write_tif(tmp_path / "vv.tif", filled(-12.0))

    valid = read_sar_valid_mask([vv])

    assert valid.shape == (SIZE, SIZE)
    assert valid.dtype == bool
    assert np.all(valid)


def test_nodata_pixels_are_invalid(tmp_path):
    # the downloaded Sentinel-1 rasters declare nodata = 0.0
    array = filled(-12.0)
    array[2, 3] = 0.0
    vv = write_tif(tmp_path / "vv.tif", array, nodata=0.0)

    valid = read_sar_valid_mask([vv])

    assert not valid[2, 3]
    assert valid.sum() == SIZE * SIZE - 1


def test_non_finite_pixels_are_invalid(tmp_path):
    array = filled(-12.0)
    array[0, 0] = np.nan
    array[1, 1] = np.inf
    vv = write_tif(tmp_path / "vv.tif", array)

    valid = read_sar_valid_mask([vv])

    assert not valid[0, 0]
    assert not valid[1, 1]
    assert valid.sum() == SIZE * SIZE - 2


def test_masks_are_combined_across_polarizations(tmp_path):
    # a pixel is only usable if EVERY polarization of the scene has data there,
    # because VV-VH is meaningless when either band is missing
    vv_array = filled(-12.0)
    vv_array[1, 1] = 0.0
    vh_array = filled(-20.0)
    vh_array[5, 5] = 0.0
    vv = write_tif(tmp_path / "vv.tif", vv_array, nodata=0.0)
    vh = write_tif(tmp_path / "vh.tif", vh_array, nodata=0.0)

    valid = read_sar_valid_mask([vv, vh])

    assert not valid[1, 1]
    assert not valid[5, 5]
    assert valid.sum() == SIZE * SIZE - 2


def test_accepts_a_bare_string_path(tmp_path):
    vh = write_tif(tmp_path / "vh.tif", filled(-20.0))

    assert read_sar_valid_mask(vh).shape == (SIZE, SIZE)


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(IOError):
        read_sar_valid_mask([str(tmp_path / "does_not_exist.tif")])


def test_raises_on_empty_file_list():
    with pytest.raises(ValueError):
        read_sar_valid_mask([])
