"""Tests for the SAR-specific transect QC.

SAR shorelines skip the MNDWI threshold filter (see test_threshold_filter.py), which
left them with no threshold-based QC at all. sar_qc_keep gives them their own checks:
the water fraction of the reference buffer (a scene segmented as ~all land or ~all
water has no real interface to contour) and, optionally, a dB range for Otsu
thresholds - the SAR counterpart of settings['otsu_threshold'].
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_tools
from coastsat.SDS_transects import (
    DEFAULT_SAR_WATER_FRACTION_RANGE,
    reject_outliers,
    sar_qc_keep,
)

MNDWI = SDS_tools.SEGMENTATION_MNDWI
SAR_OTSU = SDS_tools.SEGMENTATION_SAR_OTSU
SAR_MODEL = SDS_tools.SEGMENTATION_SAR_MODEL


def make_output(methods, thresholds=None, water_fractions=None, satnames=None):
    start = pytz.utc.localize(datetime(2023, 1, 1))
    n = len(methods)
    output = {
        "dates": [start + timedelta(days=k) for k in range(n)],
        "MNDWI_threshold": list(thresholds) if thresholds else [np.nan] * n,
        "segmentation_method": list(methods),
        "satname": list(satnames) if satnames else ["S1"] * n,
    }
    if water_fractions is not None:
        output["water_fraction"] = list(water_fractions)
    return output


SETTINGS = {
    "otsu_threshold": (-0.5, 0.5),
    "max_cross_change": 40,
    "plot_fig": False,
}


# ------------------------------------------------------------------------ sar_qc_keep


def test_healthy_sar_rows_pass():
    output = make_output([SAR_MODEL, SAR_OTSU], water_fractions=[0.5, 0.3])

    assert sar_qc_keep(output, SETTINGS).tolist() == [True, True]


def test_degenerate_water_fractions_fail():
    # ~0 or ~1 means the segmentation found no land/water interface
    output = make_output(
        [SAR_MODEL, SAR_MODEL, SAR_MODEL], water_fractions=[0.001, 0.999, 0.5]
    )

    assert sar_qc_keep(output, SETTINGS).tolist() == [False, False, True]


def test_optical_rows_always_pass():
    # water_fraction is NaN for optical rows, and even a bogus value must not judge them
    output = make_output(
        [MNDWI, MNDWI], water_fractions=[np.nan, 0.999], satnames=["S2", "S2"]
    )

    assert sar_qc_keep(output, SETTINGS).tolist() == [True, True]


def test_nan_water_fractions_are_kept():
    # an empty reference buffer produces NaN; that is not evidence of a bad scene
    output = make_output([SAR_MODEL], water_fractions=[np.nan])

    assert sar_qc_keep(output, SETTINGS).tolist() == [True]


def test_outputs_without_the_column_are_kept():
    # outputs written before water_fraction existed
    output = make_output([SAR_MODEL, SAR_OTSU])

    assert sar_qc_keep(output, SETTINGS).tolist() == [True, True]


def test_the_range_can_be_tightened():
    output = make_output([SAR_MODEL, SAR_MODEL], water_fractions=[0.1, 0.5])
    settings = dict(SETTINGS, sar_water_fraction_range=(0.3, 0.7))

    assert sar_qc_keep(output, settings).tolist() == [False, True]


def test_none_disables_the_water_fraction_check():
    output = make_output([SAR_MODEL], water_fractions=[0.999])
    settings = dict(SETTINGS, sar_water_fraction_range=None)

    assert sar_qc_keep(output, settings).tolist() == [True]


def test_the_default_range_is_deliberately_loose():
    low, high = DEFAULT_SAR_WATER_FRACTION_RANGE

    assert low <= 0.05
    assert high >= 0.95


def test_db_threshold_range_is_off_by_default():
    output = make_output([SAR_OTSU], thresholds=[-40.0], water_fractions=[0.5])

    assert sar_qc_keep(output, SETTINGS).tolist() == [True]


def test_db_threshold_range_judges_otsu_rows():
    output = make_output(
        [SAR_OTSU, SAR_OTSU, SAR_MODEL],
        thresholds=[-40.0, -20.0, np.nan],
        water_fractions=[0.5, 0.5, 0.5],
    )
    settings = dict(SETTINGS, sar_otsu_threshold=(-30.0, -10.0))

    # the model row has no threshold, so the dB range cannot judge it
    assert sar_qc_keep(output, settings).tolist() == [False, True, True]


def test_older_outputs_fall_back_to_satname():
    output = make_output(
        [MNDWI, MNDWI], water_fractions=[0.999, 0.999], satnames=["S1", "S2"]
    )
    del output["segmentation_method"]

    # only the S1 row is treated as SAR
    assert sar_qc_keep(output, SETTINGS).tolist() == [False, True]


def test_outputs_without_any_key_pass():
    output = {"dates": [1, 2], "MNDWI_threshold": [np.nan, np.nan]}

    assert sar_qc_keep(output, SETTINGS).tolist() == [True, True]


# -------------------------------------------------------------------- reject_outliers


def transects(n, method, water_fractions, thresholds=None):
    output = make_output(
        [method] * n,
        thresholds=thresholds if thresholds else [np.nan] * n,
        water_fractions=water_fractions,
    )
    cross_distance = {"transect_1": np.linspace(100.0, 110.0, n)}
    return cross_distance, output


def test_degenerate_sar_scenes_are_dropped_from_the_time_series():
    n = 40
    fractions = [0.999 if k < 5 else 0.5 for k in range(n)]
    cross_distance, output = transects(n, SAR_MODEL, fractions)

    chain_dict = reject_outliers(cross_distance, output, SETTINGS)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n - 5


def test_healthy_sar_scenes_all_survive():
    cross_distance, output = transects(40, SAR_MODEL, [0.5] * 40)

    chain_dict = reject_outliers(cross_distance, output, SETTINGS)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


def test_sar_qc_runs_even_when_the_mndwi_filter_is_disabled():
    # the MNDWI range being NaN used to skip step 2 entirely
    n = 40
    fractions = [0.999 if k < 5 else 0.5 for k in range(n)]
    cross_distance, output = transects(n, SAR_MODEL, fractions)
    settings = dict(SETTINGS, otsu_threshold=(np.nan, np.nan))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n - 5


def test_sar_qc_does_not_drop_short_transects_entirely():
    # the <30-points guard belongs to the MNDWI range filter; SAR QC must not
    # start deleting whole transects of short time-series
    n = 10
    cross_distance, output = transects(n, SAR_MODEL, [0.5] * n)
    settings = dict(SETTINGS, otsu_threshold=(np.nan, np.nan))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert "transect_1" in chain_dict
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n


def test_db_range_drops_otsu_outliers_in_the_time_series():
    n = 40
    thresholds = [-40.0 if k < 5 else -20.0 for k in range(n)]
    cross_distance, output = transects(n, SAR_OTSU, [0.5] * n, thresholds=thresholds)
    settings = dict(SETTINGS, sar_otsu_threshold=(-30.0, -10.0))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == n - 5


def test_merge_output_carries_the_water_fraction_column():
    # merge_output builds its keys from the first satellite, so the column has to
    # be appended for every satellite or it is silently dropped for the others
    output = {
        "S1": {
            "dates": [1],
            "shorelines": [np.zeros((2, 2))],
            "water_fraction": [0.5],
        },
        "S2": {
            "dates": [2],
            "shorelines": [np.zeros((2, 2))],
            "water_fraction": [np.nan],
        },
    }

    merged = SDS_tools.merge_output(output)

    assert merged["water_fraction"][0] == 0.5
    assert np.isnan(merged["water_fraction"][1])
