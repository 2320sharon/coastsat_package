"""Tests for the [VV, VH, VV-VH] Sentinel-1 composite and the image Otsu runs on."""

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_shoreline, SDS_tools
from coastsat.SDS_shoreline import describe_sar_otsu_image, get_sar_otsu_image

DATE = "2023-12-03-19-15-49"
SITENAME = "site"
SIZE = 6


def polarization_stack(values):
    """(H, W, n) stack where channel k is filled with values[k]."""
    return np.stack(
        [np.full((SIZE, SIZE), value, dtype="float64") for value in values], axis=2
    )


def s1_path(polar):
    name = f"{DATE}_S1_{SITENAME}_{polar}.tif"
    return str(Path("data") / SITENAME / "S1" / polar / name)


# ------------------------------------------------------------------ build_sar_composite


def test_third_band_is_the_db_difference():
    im = polarization_stack([-12.0, -20.0])

    composite = SDS_tools.build_sar_composite(im, ["VV", "VH"])

    assert composite.shape == (SIZE, SIZE, 3)
    assert np.all(composite[:, :, 0] == -12.0)  # VV
    assert np.all(composite[:, :, 1] == -20.0)  # VH
    assert np.all(composite[:, :, 2] == 8.0)  # VV - VH


def test_band_order_follows_the_composite_bands_not_the_input():
    # VH first on the way in, but the composite is always [VV, VH, VV-VH]
    im = polarization_stack([-20.0, -12.0])

    composite = SDS_tools.build_sar_composite(im, ["VH", "VV"])

    assert np.all(composite[:, :, 0] == -12.0)
    assert np.all(composite[:, :, 1] == -20.0)
    assert np.all(composite[:, :, 2] == 8.0)


def test_extra_polarizations_are_ignored():
    im = polarization_stack([-12.0, -20.0, -30.0])

    composite = SDS_tools.build_sar_composite(im, ["VV", "VH", "HH"])

    assert composite.shape == (SIZE, SIZE, 3)
    assert np.all(composite[:, :, 2] == 8.0)


@pytest.mark.parametrize("polarizations", [["VH"], ["VV"], ["HH", "HV"]])
def test_returns_none_without_both_vv_and_vh(polarizations):
    im = polarization_stack([-20.0] * len(polarizations))

    assert SDS_tools.build_sar_composite(im, polarizations) is None


def test_composite_bands_are_named():
    assert SDS_tools.SAR_COMPOSITE_BANDS == ("VV", "VH", "VV-VH")


# ------------------------------------------------------------------ get_sar_otsu_image


def test_thresholds_the_mean_of_vv_and_vh():
    composite = SDS_tools.build_sar_composite(
        polarization_stack([-12.0, -20.0]), ["VV", "VH"]
    )

    im_sar = get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")])

    assert np.all(im_sar == -16.0)  # (-12 + -20) / 2


def test_the_difference_band_is_excluded_from_the_mean():
    # mean([VV, VH, VV-VH]) == (2/3)*VV, which would cancel VH entirely and make the
    # threshold equivalent to VV alone. Guard against anyone "simplifying" to np.mean.
    composite = SDS_tools.build_sar_composite(
        polarization_stack([-12.0, -20.0]), ["VV", "VH"]
    )

    im_sar = get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")])

    assert not np.allclose(im_sar, np.mean(composite, axis=2))
    assert not np.allclose(im_sar, composite[:, :, 0])


def test_vh_changes_the_threshold_input():
    # a different VH must move the Otsu input; this fails if VH is being cancelled out
    a = SDS_tools.build_sar_composite(polarization_stack([-12.0, -20.0]), ["VV", "VH"])
    b = SDS_tools.build_sar_composite(polarization_stack([-12.0, -26.0]), ["VV", "VH"])
    fn = [s1_path("VV"), s1_path("VH")]

    assert not np.allclose(get_sar_otsu_image(a, fn), get_sar_otsu_image(b, fn))


def test_single_polarization_thresholds_that_band():
    im = polarization_stack([-20.0])

    assert np.all(get_sar_otsu_image(im, [s1_path("VH")]) == -20.0)


def test_accepts_a_bare_string_path():
    im = polarization_stack([-20.0])

    assert np.all(get_sar_otsu_image(im, s1_path("VH")) == -20.0)


def test_result_is_two_dimensional():
    composite = SDS_tools.build_sar_composite(
        polarization_stack([-12.0, -20.0]), ["VV", "VH"]
    )

    # otsu_threshold and find_shoreline_SAR both require a 2D array
    assert get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")]).ndim == 2


def test_extra_polarizations_do_not_enter_the_mean():
    # VV+VH+HH: the mean is still VV and VH only
    im = polarization_stack([-12.0, -20.0, -30.0])
    fn = [s1_path("VV"), s1_path("VH"), s1_path("HH")]

    assert np.all(get_sar_otsu_image(im, fn) == -16.0)


# ------------------------------------------------------------------ no smoothing


def test_otsu_image_is_not_smoothed():
    # the SAR imagery is thresholded as acquired: no median/speckle filter, so a
    # single outlying pixel must survive into the image Otsu sees
    im = polarization_stack([-12.0, -20.0])
    im[3, 3, 0] = 40.0  # a lone bright pixel in VV
    composite = SDS_tools.build_sar_composite(im, ["VV", "VH"])

    im_sar = get_sar_otsu_image(composite, [s1_path("VV"), s1_path("VH")])

    assert im_sar[3, 3] == pytest.approx((40.0 + -20.0) / 2)
    # a median filter would have replaced it with the surrounding value
    assert im_sar[3, 3] != pytest.approx(-16.0)


def test_extract_shorelines_does_not_filter_the_sar_image():
    # guards against a median/gaussian filter being reintroduced in the S1 branch
    source = inspect.getsource(SDS_shoreline.extract_shorelines)

    for smoother in ("median_filter", "gaussian_filter", "uniform_filter", "convolve"):
        assert smoother not in source, f"{smoother} reintroduced into extract_shorelines"


# ------------------------------------------------------------------ logging


@pytest.mark.parametrize(
    "polarizations,expected",
    [
        (["VV", "VH"], "the mean of VV and VH"),
        (["VV", "VH", "HH"], "the mean of VV and VH"),
        (["VH"], "VH"),
        (["VV"], "VV"),
    ],
)
def test_describe_matches_what_is_thresholded(polarizations, expected):
    assert describe_sar_otsu_image([s1_path(p) for p in polarizations]) == expected
