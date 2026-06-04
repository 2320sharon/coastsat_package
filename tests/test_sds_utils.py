"""Credential-free unit tests for pure CoastSat helper functions.

These exercise deterministic utilities — coordinate transforms, the normalised
difference index, cloud-cover fraction, finite-data filtering and ordinal
formatting — that need no Earth Engine auth, network access or file I/O, so they
run anywhere pytest does (locally and in CI). The root-level ``test_*.py``
modules, by contrast, require Google Earth Engine.
"""

import numpy as np
import pytest

from coastsat.SDS_tools import (
    convert_pix2world,
    convert_world2pix,
    nd_index,
    ordinal,
)
from coastsat.SDS_preprocess import calculate_cloud_cover_combined
from coastsat.SDS_shoreline import get_finite_data


# North-up geotransform [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]:
#   X = 300000 + 10*col,  Y = 6000000 - 10*row
GEOREF = np.array([300000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])


class TestCoordinateTransforms:
    def test_pix2world_known_affine(self):
        # pixel (row=5, col=3) -> world (X=300030, Y=5999950)
        world = convert_pix2world(np.array([[5.0, 3.0]]), GEOREF)
        np.testing.assert_allclose(world, [[300030.0, 5999950.0]])

    def test_world2pix_known_affine(self):
        # world (X=300030, Y=5999950) -> (col=3, row=5); world2pix returns [col, row]
        pix = convert_world2pix(np.array([[300030.0, 5999950.0]]), GEOREF)
        np.testing.assert_allclose(pix, [[3.0, 5.0]], atol=1e-9)

    def test_pix2world_accepts_list_of_arrays(self):
        out = convert_pix2world([np.array([[5.0, 3.0]])], GEOREF)
        assert isinstance(out, list)
        np.testing.assert_allclose(out[0], [[300030.0, 5999950.0]])

    def test_pix2world_rejects_bad_type(self):
        with pytest.raises(Exception):
            convert_pix2world("not an array", GEOREF)


class TestNdIndex:
    def test_normalised_difference_values(self):
        im1 = np.array([[3.0, 1.0], [5.0, 2.0]])
        im2 = np.array([[1.0, 1.0], [5.0, 0.0]])
        cloud = np.zeros((2, 2), dtype=bool)
        nd = nd_index(im1, im2, cloud)
        # (im1 - im2) / (im1 + im2) element-wise
        np.testing.assert_allclose(nd, [[0.5, 0.0], [0.0, 1.0]])

    def test_cloud_pixels_are_nan(self):
        im1 = np.array([[3.0, 1.0], [5.0, 2.0]])
        im2 = np.array([[1.0, 1.0], [5.0, 0.0]])
        cloud = np.array([[True, False], [False, False]])
        nd = nd_index(im1, im2, cloud)
        assert np.isnan(nd[0, 0])           # cloud pixel masked out
        np.testing.assert_allclose(nd[1, 1], 1.0)  # clear pixel computed


class TestCloudCover:
    def test_no_clouds(self):
        assert calculate_cloud_cover_combined(np.zeros((4, 4), dtype=bool)) == 0.0

    def test_all_clouds(self):
        assert calculate_cloud_cover_combined(np.ones((4, 4), dtype=bool)) == 1.0

    def test_quarter_clouds(self):
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        assert calculate_cloud_cover_combined(mask) == 0.25

    def test_none_returns_zero(self):
        assert calculate_cloud_cover_combined(None) == 0.0

    def test_empty_returns_zero(self):
        assert calculate_cloud_cover_combined(np.array([], dtype=bool)) == 0.0


class TestGetFiniteData:
    def test_filters_nan_and_inf(self):
        data = np.array([1.0, np.nan, 2.0, np.inf, -np.inf, 3.0])
        np.testing.assert_array_equal(get_finite_data(data), [1.0, 2.0, 3.0])

    def test_all_finite_unchanged(self):
        data = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(get_finite_data(data), [1.0, 2.0, 3.0])

    def test_raises_when_no_finite_values(self):
        with pytest.raises(ValueError):
            get_finite_data(np.array([np.nan, np.inf]))


@pytest.mark.parametrize(
    "n, expected",
    [
        (1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"),
        (11, "11th"), (12, "12th"), (13, "13th"),
        (21, "21st"), (22, "22nd"), (23, "23rd"),
        (100, "100th"), (101, "101st"),
        (111, "111th"), (112, "112th"), (113, "113th"),
    ],
)
def test_ordinal(n, expected):
    assert ordinal(n) == expected
