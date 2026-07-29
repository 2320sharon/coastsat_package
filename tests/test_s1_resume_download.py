"""Tests that resuming a download completes scenes missing a polarization.

A site first downloaded as VH-only must not stay single-polarization forever: matching
on acquisition date alone made every existing scene look complete, so the missing VV was
never fetched and the scene could never reach the segmentation model.
"""

from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat.SDS_download import (
    get_downloaded_sar_polarizations,
    get_incomplete_sar_dates,
    remove_existing_imagery,
)

SITENAME = "site"
DATES = [
    pytz.utc.localize(datetime(2023, 12, 3, 19, 15, 49)),
    pytz.utc.localize(datetime(2023, 12, 6, 8, 39, 48)),
]
STAMPS = ["2023-12-03-19-15-49", "2023-12-06-08-39-48"]
GEOTRANSFORM = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0]


def write_tif(path):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), 4, 4, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    dataset.GetRasterBand(1).WriteArray(np.full((4, 4), -20.0))
    dataset = None


def build_session(tmp_path, polarizations_per_scene):
    """Create S1/<POL>/ folders holding the given polarizations for each scene."""
    s1 = tmp_path / SITENAME / "S1"
    for polars in polarizations_per_scene:
        for polar in polars:
            (s1 / polar).mkdir(parents=True, exist_ok=True)
    for index, polars in enumerate(polarizations_per_scene):
        for polar in polars:
            write_tif(s1 / polar / f"{STAMPS[index]}_S1_{SITENAME}_{polar}.tif")
    return {
        "filepath": str(tmp_path),
        "sitename": SITENAME,
        "sat_list": ["S1"],
    }


def metadata_s1_for(polarizations_per_scene):
    """Just the 'S1' entry, which is what the two helpers take."""
    return {
        "dates": list(DATES[: len(polarizations_per_scene)]),
        "filenames": [
            f"{STAMPS[i]}_S1_{SITENAME}_{polars[0]}.tif"
            for i, polars in enumerate(polarizations_per_scene)
        ],
        "band_order": [list(polars) for polars in polarizations_per_scene],
    }


def metadata_for(polarizations_per_scene):
    """The full metadata dict, which is what remove_existing_imagery takes."""
    return {"S1": metadata_s1_for(polarizations_per_scene)}


def image_dict_for(dates):
    return {
        "S1": [
            {"properties": {"system:time_start": date.timestamp() * 1000}}
            for date in dates
        ]
    }


# ------------------------------------------------------- get_downloaded_sar_polarizations


def test_reads_the_polarizations_from_disk(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"], ["VH"]])

    by_date = get_downloaded_sar_polarizations(
        inputs, metadata_s1_for([["VV", "VH"], ["VH"]])
    )

    assert by_date[DATES[0]] == {"VV", "VH"}
    assert by_date[DATES[1]] == {"VH"}


def test_a_deleted_file_makes_the_scene_incomplete(tmp_path):
    # the metadata still claims VV+VH, but the VV .tif is gone
    inputs = build_session(tmp_path, [["VV", "VH"]])
    (tmp_path / SITENAME / "S1" / "VV" / f"{STAMPS[0]}_S1_{SITENAME}_VV.tif").unlink()

    by_date = get_downloaded_sar_polarizations(inputs, metadata_s1_for([["VV", "VH"]]))

    assert by_date[DATES[0]] == {"VH"}


def test_falls_back_to_band_order_when_files_are_unreadable(tmp_path):
    inputs = {"filepath": str(tmp_path / "nothing"), "sitename": SITENAME}

    by_date = get_downloaded_sar_polarizations(inputs, metadata_s1_for([["VV", "VH"]]))

    assert by_date[DATES[0]] == {"VV", "VH"}


# ------------------------------------------------------------- get_incomplete_sar_dates


def test_vh_only_scene_is_incomplete_for_a_vv_vh_request(tmp_path):
    inputs = build_session(tmp_path, [["VH"]])

    incomplete = get_incomplete_sar_dates(
        inputs, metadata_s1_for([["VH"]]), ["VV", "VH"]
    )

    assert incomplete == {DATES[0]}


def test_a_complete_scene_is_not_flagged(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"]])

    incomplete = get_incomplete_sar_dates(
        inputs, metadata_s1_for([["VV", "VH"]]), ["VV", "VH"]
    )

    assert incomplete == set()


def test_vh_only_scene_is_complete_for_a_vh_request(tmp_path):
    # asking only for VH must not re-download a VH-only site
    inputs = build_session(tmp_path, [["VH"]])

    incomplete = get_incomplete_sar_dates(inputs, metadata_s1_for([["VH"]]), ["VH"])

    assert incomplete == set()


def test_extra_polarizations_on_disk_do_not_make_a_scene_incomplete(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"]])

    incomplete = get_incomplete_sar_dates(inputs, metadata_s1_for([["VV", "VH"]]), ["VH"])

    assert incomplete == set()


# ------------------------------------------------------------- remove_existing_imagery


def test_resume_redownloads_the_scene_missing_vv(tmp_path):
    # the regression: this scene used to be skipped and stay VH-only forever
    inputs = build_session(tmp_path, [["VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }
    image_dict = image_dict_for(DATES[:1])

    result = remove_existing_imagery(
        image_dict, metadata_for([["VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1, "the incomplete scene was not queued for download"


def test_resume_skips_a_complete_scene(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }

    result = remove_existing_imagery(
        image_dict_for(DATES[:1]), metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 0


def test_resume_downloads_only_the_incomplete_scene(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"], ["VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }

    result = remove_existing_imagery(
        image_dict_for(DATES), metadata_for([["VV", "VH"], ["VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1
    kept = datetime.fromtimestamp(
        result["S1"][0]["properties"]["system:time_start"] / 1000, tz=pytz.utc
    )
    assert kept == DATES[1], "the wrong scene was queued"


def test_inputs_is_required(tmp_path):
    # inputs carries the requested polarizations, so without it an incomplete scene
    # cannot be told from a complete one. Failing loudly beats silently leaving a site
    # single-polarization forever, so this must raise rather than fall back.
    with pytest.raises(TypeError):
        remove_existing_imagery(
            image_dict_for(DATES[:1]), metadata_for([["VH"]]), ["S1"]
        )


def test_a_site_with_no_sentinel_1_properties_uses_the_default_polarizations(tmp_path):
    # inputs need not name the polarizations; the VV+VH default still applies, so a
    # legacy VH-only site is completed even by a caller that sets nothing
    inputs = build_session(tmp_path, [["VH"]])

    result = remove_existing_imagery(
        image_dict_for(DATES[:1]), metadata_for([["VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1


def test_optical_satellites_are_unaffected(tmp_path):
    metadata = {"L9": {"dates": DATES[:1], "filenames": ["a.tif"]}}
    image_dict = {
        "L9": [
            {"properties": {"system:time_start": date.timestamp() * 1000}}
            for date in DATES
        ]
    }

    result = remove_existing_imagery(image_dict, metadata, ["L9"], {"sitename": "s"})

    assert len(result["L9"]) == 1  # only the second date is new


def test_s1_and_an_optical_satellite_in_one_call(tmp_path):
    # the S1 completeness check must not disturb the date-only filtering of the others
    inputs = build_session(tmp_path, [["VH"]])
    inputs["sat_list"] = ["S1", "L9"]
    metadata = metadata_for([["VH"]])
    metadata["L9"] = {"dates": DATES[:1], "filenames": ["a.tif"]}
    image_dict = image_dict_for(DATES)
    image_dict["L9"] = [
        {"properties": {"system:time_start": date.timestamp() * 1000}}
        for date in DATES
    ]

    result = remove_existing_imagery(image_dict, metadata, ["S1", "L9"], inputs)

    assert len(result["S1"]) == 2  # the incomplete scene, plus the genuinely new one
    assert len(result["L9"]) == 1  # only the second date is new


def test_a_satellite_with_no_metadata_keeps_every_image(tmp_path):
    # nothing downloaded yet: the site has no S1 entry at all
    inputs = build_session(tmp_path, [])

    result = remove_existing_imagery(image_dict_for(DATES), {}, ["S1"], inputs)

    assert len(result["S1"]) == 2


def test_a_satellite_with_no_downloaded_dates_keeps_every_image(tmp_path):
    inputs = build_session(tmp_path, [])
    metadata = {"S1": {"dates": [], "filenames": [], "band_order": []}}

    result = remove_existing_imagery(image_dict_for(DATES), metadata, ["S1"], inputs)

    assert len(result["S1"]) == 2


def test_no_available_images_leaves_the_dict_empty(tmp_path):
    inputs = build_session(tmp_path, [["VV", "VH"]])

    result = remove_existing_imagery(
        {"S1": []}, metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    assert result["S1"] == []


def test_a_polarization_that_was_never_downloaded_requeues_the_scene(tmp_path):
    # asking for HH on a VV+VH site: the scene is incomplete for that request
    inputs = build_session(tmp_path, [["VV", "VH"]])
    inputs["sentinel_1_properties"] = {"transmitterReceiverPolarisation": ["HH"]}

    result = remove_existing_imagery(
        image_dict_for(DATES[:1]), metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    assert len(result["S1"]) == 1


def test_it_reports_how_many_scenes_are_incomplete(tmp_path, capsys):
    inputs = build_session(tmp_path, [["VH"], ["VH"]])
    inputs["sentinel_1_properties"] = {
        "transmitterReceiverPolarisation": ["VV", "VH"]
    }

    remove_existing_imagery(
        image_dict_for(DATES), metadata_for([["VH"], ["VH"]]), ["S1"], inputs
    )

    printed = capsys.readouterr().out
    assert "2 existing images are missing one of VV+VH" in printed


def test_a_complete_site_reports_nothing_to_download(tmp_path, capsys):
    inputs = build_session(tmp_path, [["VV", "VH"]])

    remove_existing_imagery(
        image_dict_for(DATES[:1]), metadata_for([["VV", "VH"]]), ["S1"], inputs
    )

    printed = capsys.readouterr().out
    assert "missing one of" not in printed
    assert "1 images already exist, 0 to download" in printed
