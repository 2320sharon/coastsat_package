"""Tests that the otsu_threshold QC filter cannot silently delete SAR shorelines.

output['MNDWI_threshold'] holds a different quantity depending on how the shoreline was
mapped: an MNDWI value (optical), an Otsu threshold in dB (SAR), or NaN (SAR segmentation
model, which has no threshold). Comparing the latter two against an MNDWI range drops
them - NaN fails every comparison and a dB value near -20 is outside any sensible MNDWI
range - so they have to skip the filter instead.
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_tools, SDS_transects
from coastsat.SDS_transects import reject_outliers, threshold_filter_applies

MNDWI = SDS_tools.SEGMENTATION_MNDWI
SAR_OTSU = SDS_tools.SEGMENTATION_SAR_OTSU
SAR_MODEL = SDS_tools.SEGMENTATION_SAR_MODEL

# wide enough that no optical shoreline is rejected on its own merits
OPTICAL_RANGE = (-0.5, 0.5)


def make_output(methods, thresholds, satnames=None):
    start = pytz.utc.localize(datetime(2023, 1, 1))
    return {
        "dates": [start + timedelta(days=k) for k in range(len(methods))],
        "MNDWI_threshold": list(thresholds),
        "segmentation_method": list(methods),
        "satname": list(satnames) if satnames else ["S2"] * len(methods),
    }


# ------------------------------------------------------------ threshold_filter_applies


def test_optical_shorelines_are_filterable():
    output = make_output([MNDWI, MNDWI], [-0.1, 0.2])

    assert threshold_filter_applies(output).tolist() == [True, True]


def test_model_segmented_shorelines_are_not_filterable():
    # the whole point: a NaN threshold must not be compared against a range
    output = make_output([MNDWI, SAR_MODEL], [-0.1, np.nan])

    assert threshold_filter_applies(output).tolist() == [True, False]


def test_sar_otsu_shorelines_are_not_filterable():
    # its threshold is in dB, which no MNDWI range can meaningfully bound
    output = make_output([MNDWI, SAR_OTSU], [-0.1, -19.3])

    assert threshold_filter_applies(output).tolist() == [True, False]


def test_a_nan_threshold_is_never_filterable():
    # even if something mislabels the method, NaN can never satisfy a range
    output = make_output([MNDWI], [np.nan])

    assert threshold_filter_applies(output).tolist() == [False]


def test_falls_back_to_satname_for_older_outputs():
    # outputs written before segmentation_method existed
    output = make_output([MNDWI, SAR_OTSU], [-0.1, -19.3], satnames=["S2", "S1"])
    del output["segmentation_method"]

    assert threshold_filter_applies(output).tolist() == [True, False]


def test_optical_only_outputs_without_either_key_are_unchanged():
    output = {"MNDWI_threshold": [-0.1, 0.2], "dates": [None, None]}

    assert threshold_filter_applies(output).tolist() == [True, True]


# ------------------------------------------------------------------- reject_outliers


def transects(n, method, threshold, satname="S1"):
    """n shorelines, all mapped the same way, with a clean linear chainage."""
    output = make_output([method] * n, [threshold] * n, satnames=[satname] * n)
    # a gently varying series so the despiking step keeps everything
    cross_distance = {"transect_1": np.linspace(100.0, 110.0, n)}
    return cross_distance, output


SETTINGS = {
    "otsu_threshold": OPTICAL_RANGE,
    "max_cross_change": 40,
    "plot_fig": False,
}


def test_model_segmented_shorelines_survive_the_filter():
    cross_distance, output = transects(40, SAR_MODEL, np.nan)

    chain_dict = reject_outliers(cross_distance, output, SETTINGS)

    assert "transect_1" in chain_dict, "every SAR shoreline was dropped by the QC filter"
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


def test_sar_otsu_shorelines_survive_the_filter():
    cross_distance, output = transects(40, SAR_OTSU, -19.3)

    chain_dict = reject_outliers(cross_distance, output, SETTINGS)

    assert "transect_1" in chain_dict
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


def test_optical_shorelines_outside_the_range_are_still_rejected():
    # the filter must keep doing its job where it is meaningful
    cross_distance, output = transects(40, MNDWI, 0.9, satname="S2")

    chain_dict = reject_outliers(cross_distance, output, SETTINGS)

    # fewer than 30 survive the range, so the transect is dropped entirely
    assert "transect_1" not in chain_dict


def test_optical_shorelines_inside_the_range_are_kept():
    cross_distance, output = transects(40, MNDWI, -0.1, satname="S2")

    chain_dict = reject_outliers(cross_distance, output, SETTINGS)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


def test_a_mixed_time_series_keeps_sar_and_filters_optical():
    n = 60
    methods = [MNDWI if k % 2 else SAR_MODEL for k in range(n)]
    # optical thresholds are out of range, SAR is NaN
    thresholds = [0.9 if k % 2 else np.nan for k in range(n)]
    satnames = ["S2" if k % 2 else "S1" for k in range(n)]
    output = make_output(methods, thresholds, satnames=satnames)
    cross_distance = {"transect_1": np.linspace(100.0, 110.0, n)}

    chain_dict = reject_outliers(cross_distance, output, dict(SETTINGS))

    # the 30 SAR shorelines survive, the 30 out-of-range optical ones do not
    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 30


def test_a_nan_otsu_threshold_setting_still_disables_the_filter():
    cross_distance, output = transects(40, MNDWI, 0.9, satname="S2")
    settings = dict(SETTINGS, otsu_threshold=(np.nan, np.nan))

    chain_dict = reject_outliers(cross_distance, output, settings)

    assert np.sum(~np.isnan(chain_dict["transect_1"])) == 40


# --------------------------------------------------------------- the output column


def test_segmentation_method_constants_are_distinct():
    assert len({MNDWI, SAR_OTSU, SAR_MODEL}) == 3


def test_merge_output_carries_the_new_column():
    # merge_output builds its keys from the first satellite, so the column has to be
    # present for every satellite or it is silently dropped for the others
    output = {
        "S1": {
            "dates": [1],
            "shorelines": [np.zeros((2, 2))],
            "MNDWI_threshold": [np.nan],
            "segmentation_method": [SAR_MODEL],
        },
        "S2": {
            "dates": [2],
            "shorelines": [np.zeros((2, 2))],
            "MNDWI_threshold": [-0.1],
            "segmentation_method": [MNDWI],
        },
    }

    merged = SDS_tools.merge_output(output)

    assert merged["segmentation_method"] == [SAR_MODEL, MNDWI]
    assert merged["satname"] == ["S1", "S2"]
