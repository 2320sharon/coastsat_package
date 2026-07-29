"""Tests for reading the polarization files of a SAR scene into one stacked array."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat.SDS_tools import read_sar_image

GEOTRANSFORM = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0]


def write_tif(path, value, size=8, geotransform=None):
    """Write a small single-band Float64 .tif, like an Earth Engine S1 download."""
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geotransform or GEOTRANSFORM)
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype="float64"))
    dataset = None
    return str(path)


def test_stacks_polarizations_in_the_order_given(tmp_path):
    vv = write_tif(tmp_path / "vv.tif", -12.0)
    vh = write_tif(tmp_path / "vh.tif", -20.0)

    im, georef = read_sar_image([vv, vh])

    assert im.shape == (8, 8, 2)
    assert np.all(im[:, :, 0] == -12.0)
    assert np.all(im[:, :, 1] == -20.0)
    assert np.allclose(georef, GEOTRANSFORM)


def test_reversing_the_file_list_reverses_the_channels(tmp_path):
    vv = write_tif(tmp_path / "vv.tif", -12.0)
    vh = write_tif(tmp_path / "vh.tif", -20.0)

    im, _ = read_sar_image([vh, vv])

    assert np.all(im[:, :, 0] == -20.0)
    assert np.all(im[:, :, 1] == -12.0)


def test_single_file_returns_one_channel(tmp_path):
    # a scene downloaded before multi-polarization support
    vh = write_tif(tmp_path / "vh.tif", -20.0)

    im, _ = read_sar_image([vh])

    assert im.shape == (8, 8, 1)
    assert np.all(im[:, :, 0] == -20.0)


def test_accepts_a_bare_string_path(tmp_path):
    vh = write_tif(tmp_path / "vh.tif", -20.0)

    im, _ = read_sar_image(vh)

    assert im.shape == (8, 8, 1)


def test_raises_when_shapes_differ(tmp_path):
    vv = write_tif(tmp_path / "vv.tif", -12.0, size=8)
    vh = write_tif(tmp_path / "vh.tif", -20.0, size=6)

    with pytest.raises(ValueError, match="same grid"):
        read_sar_image([vv, vh])


def test_raises_when_geotransforms_differ(tmp_path):
    vv = write_tif(tmp_path / "vv.tif", -12.0)
    vh = write_tif(
        tmp_path / "vh.tif",
        -20.0,
        geotransform=[600000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0],
    )

    with pytest.raises(ValueError, match="same grid"):
        read_sar_image([vv, vh])


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(IOError):
        read_sar_image([str(tmp_path / "does_not_exist.tif")])


def test_raises_on_empty_file_list():
    with pytest.raises(ValueError):
        read_sar_image([])
