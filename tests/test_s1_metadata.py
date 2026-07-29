"""Tests for the Sentinel-1 metadata files: one per scene, recording the band order."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat.SDS_download import (
    get_band_order_from_meta,
    get_metadata,
    read_metadata_file,
    write_metadata_file,
)

DATE = "2023-12-03-19-15-49"
OTHER_DATE = "2023-12-06-08-39-48"
SITENAME = "site"


def s1_metadict(polarizations, date=DATE):
    return {
        "filename": f"{date}_S1_{SITENAME}_{polarizations[0]}.tif",
        "epsg": 32756,
        "im_width": 168,
        "im_height": 436,
        "band_order": polarizations,
        "orbitProperties_pass": "DESCENDING",
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "resolution": "H",
        "resolution_meters": 10,
        "instrumentMode": "IW",
    }


def legacy_metadict(date=OTHER_DATE):
    """A metadata file as written before multi-polarization support."""
    return {
        "filename": f"{date}_S1_{SITENAME}_VH.tif",
        "epsg": 32756,
        "im_width": 168,
        "im_height": 436,
        "orbitProperties_pass": "DESCENDING",
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "saved_polarization": "VH",
        "resolution": "H",
        "resolution_meters": 10,
        "instrumentMode": "IW",
    }


def make_meta_dir(root, satname="S1"):
    meta = root / SITENAME / satname / "meta"
    meta.mkdir(parents=True)
    return meta


def inputs_for(root, sat_list=("S1",)):
    return {
        "filepath": str(root),
        "sitename": SITENAME,
        "sat_list": list(sat_list),
        "dates": ["2020-01-01", "2030-01-01"],
    }


# ------------------------------------------------------------------ write / read


def test_written_without_polar_has_no_suffix(tmp_path):
    write_metadata_file(str(tmp_path), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))

    assert (tmp_path / f"{DATE}_S1_{SITENAME}.txt").exists()


def test_legacy_polar_suffix_still_available(tmp_path):
    write_metadata_file(
        str(tmp_path), f"{DATE}_S1_{SITENAME}", legacy_metadict(), polar="VH"
    )

    assert (tmp_path / f"{DATE}_S1_{SITENAME}_VH.txt").exists()


def test_band_order_round_trips_as_a_list(tmp_path):
    write_metadata_file(str(tmp_path), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))

    meta_info = read_metadata_file(str(tmp_path / f"{DATE}_S1_{SITENAME}.txt"))

    assert meta_info["band_order"] == ["VV", "VH"]


# ------------------------------------------------------------------ get_band_order_from_meta


def test_band_order_is_preferred():
    assert get_band_order_from_meta({"band_order": ["VV", "VH"]}) == ["VV", "VH"]


def test_falls_back_to_saved_polarization():
    assert get_band_order_from_meta({"saved_polarization": "VH"}) == ["VH"]


def test_returns_empty_when_neither_is_recorded():
    assert get_band_order_from_meta({}) == []


def test_accepts_a_comma_separated_band_order():
    assert get_band_order_from_meta({"band_order": "VV, VH"}) == ["VV", "VH"]


# ------------------------------------------------------------------ get_metadata


def test_get_metadata_populates_band_order(tmp_path):
    meta = make_meta_dir(tmp_path)
    write_metadata_file(str(meta), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))

    metadata = get_metadata(inputs_for(tmp_path))

    assert metadata["S1"]["band_order"] == [["VV", "VH"]]


def test_one_scene_produces_one_entry(tmp_path):
    # a dual-polarization scene gets ONE metadata file, so it must not be listed twice
    meta = make_meta_dir(tmp_path)
    write_metadata_file(str(meta), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))
    write_metadata_file(
        str(meta), f"{OTHER_DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"], OTHER_DATE)
    )

    metadata = get_metadata(inputs_for(tmp_path))

    assert len(metadata["S1"]["filenames"]) == 2
    assert metadata["S1"]["filenames"] == [
        f"{DATE}_S1_{SITENAME}_VV.tif",
        f"{OTHER_DATE}_S1_{SITENAME}_VV.tif",
    ]


def test_band_order_is_index_aligned_with_filenames(tmp_path):
    # a session holding both a new dual-pol scene and a legacy VH-only one
    meta = make_meta_dir(tmp_path)
    write_metadata_file(str(meta), f"{DATE}_S1_{SITENAME}", s1_metadict(["VV", "VH"]))
    write_metadata_file(
        str(meta), f"{OTHER_DATE}_S1_{SITENAME}", legacy_metadict(), polar="VH"
    )

    metadata = get_metadata(inputs_for(tmp_path))

    assert len(metadata["S1"]["band_order"]) == len(metadata["S1"]["filenames"])
    assert metadata["S1"]["band_order"] == [["VV", "VH"], ["VH"]]


def test_optical_metadata_has_no_band_order(tmp_path):
    meta = make_meta_dir(tmp_path, satname="S2")
    write_metadata_file(
        str(meta),
        f"{DATE}_S2_{SITENAME}",
        {
            "filename": f"{DATE}_S2_{SITENAME}_ms.tif",
            "epsg": 32756,
            "acc_georef": -1,
            "image_quality": "PASSED",
            "im_width": 100,
            "im_height": 100,
        },
    )

    metadata = get_metadata(inputs_for(tmp_path, sat_list=("S2",)))

    assert "band_order" not in metadata["S2"]
