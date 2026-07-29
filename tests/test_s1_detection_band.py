"""Tests that SAR shoreline detection still thresholds VH once VV is also downloaded.

im_ms now holds one channel per polarization, so the channel to threshold is chosen
by polarization name rather than by position.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_tools
from coastsat.SDS_shoreline import (
    SAR_DETECTION_POLARIZATION,
    describe_sar_otsu_image,
    get_sar_band_index,
    get_sar_detection_polarization,
    get_sar_otsu_image,
    warn_unknown_sar_detection_polarization,
)

DATE = "2023-12-03-19-15-49"


def s1_path(polar, suffix=""):
    return str(Path("data") / "site" / "S1" / polar / f"{DATE}_S1_site_{polar}{suffix}.tif")


def polarization_stack(values, size=4):
    """An (H, W, n) stack with one constant band per value."""
    return np.stack([np.full((size, size), v, dtype="float64") for v in values], axis=2)


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(str(message))


def test_default_polarization_is_vh():
    assert SAR_DETECTION_POLARIZATION == "VH"


@pytest.mark.parametrize(
    "fn,expected,reason",
    [
        ([s1_path("VV"), s1_path("VH")], 1, "VV+VH: VH is the second channel"),
        ([s1_path("VH"), s1_path("VV")], 0, "reversed: VH is the first channel"),
        ([s1_path("VH")], 0, "legacy VH-only scene"),
        ([s1_path("VV")], 0, "VV-only request falls back to the only channel"),
        ([s1_path("HH"), s1_path("HV")], 0, "no VH present, falls back to channel 0"),
    ],
)
def test_get_sar_band_index(fn, expected, reason):
    assert get_sar_band_index(fn) == expected, reason


def test_handles_duplicate_named_files():
    fn = [s1_path("VV", "_dup0"), s1_path("VH", "_dup0")]

    assert get_sar_band_index(fn) == 1


def test_accepts_a_bare_string_path():
    assert get_sar_band_index(s1_path("VH")) == 0


def test_polarization_is_configurable():
    fn = [s1_path("VV"), s1_path("VH")]

    assert get_sar_band_index(fn, polarization="VV") == 0


# ------------------------------------------- settings['sar_detection_polarization']


DUAL_POL = [s1_path("VV"), s1_path("VH")]
# a composite, which is what preprocess_image hands over for a dual-polarization scene
COMPOSITE = polarization_stack([-12.0, -20.0, 8.0])


@pytest.mark.parametrize(
    "settings,expected",
    [
        (None, None),
        ({}, None),
        ({"sar_detection_polarization": None}, None),
        ({"sar_detection_polarization": ""}, None),
        ({"sar_detection_polarization": "vv"}, "VV"),
        ({"sar_detection_polarization": " VH "}, "VH"),
    ],
)
def test_get_sar_detection_polarization(settings, expected):
    assert get_sar_detection_polarization(settings) == expected


def test_the_default_is_unchanged_when_the_setting_is_absent():
    # dual-polarization still thresholds the mean of VV and VH, as it always has
    assert np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, {}) == -16.0)
    assert describe_sar_otsu_image(DUAL_POL, {}) == "the mean of VV and VH"


def test_the_setting_overrides_the_mean_of_vv_and_vh():
    settings = {"sar_detection_polarization": "VV"}

    assert np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, settings) == -12.0)
    assert describe_sar_otsu_image(DUAL_POL, settings) == "VV"


def test_the_setting_can_select_vh_from_a_dual_polarization_scene():
    settings = {"sar_detection_polarization": "VH"}

    assert np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, settings) == -20.0)


def test_the_setting_selects_a_band_of_a_single_polarization_scene():
    im = polarization_stack([-12.0, -30.0])
    fn = [s1_path("VV"), s1_path("HH")]  # no VH, so no composite was built

    settings = {"sar_detection_polarization": "HH"}

    assert np.all(get_sar_otsu_image(im, fn, settings) == -30.0)
    assert describe_sar_otsu_image(fn, settings) == "HH"


def test_a_band_the_scene_does_not_carry_falls_back_to_the_default():
    # one odd scene in a run must still map rather than raising
    settings = {"sar_detection_polarization": "HV"}

    assert np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, settings) == -16.0)
    # the log reports the band actually thresholded, not the one that was asked for
    assert describe_sar_otsu_image(DUAL_POL, settings) == "the mean of VV and VH"


def test_the_difference_band_cannot_be_selected_by_a_polarization_name():
    # VV-VH is a derived band, not a polarization, so no setting value reaches it
    for polarization in SDS_tools.SAR_POLARIZATIONS:
        settings = {"sar_detection_polarization": polarization}
        assert not np.all(get_sar_otsu_image(COMPOSITE, DUAL_POL, settings) == 8.0)


def test_an_unknown_polarization_is_warned_about_once():
    logger = RecordingLogger()

    warn_unknown_sar_detection_polarization(
        {"sar_detection_polarization": "VVH"}, logger=logger
    )

    assert len(logger.warnings) == 1
    assert "VVH" in logger.warnings[0]


@pytest.mark.parametrize("settings", [{}, {"sar_detection_polarization": "VV"}])
def test_no_warning_for_the_default_or_a_real_polarization(settings):
    logger = RecordingLogger()

    warn_unknown_sar_detection_polarization(settings, logger=logger)

    assert logger.warnings == []
