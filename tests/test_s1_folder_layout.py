"""Tests for the Sentinel-1 folder layout: one folder per polarization.

Covers creating the folders, listing them back, and pairing up the files of a scene -
including sessions downloaded before multi-polarization support, which have VH only.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat import SDS_tools

DATE = "2023-12-03-19-15-49"
SITENAME = "site"


def write_tif(path, value=1.0, size=8, geotransform=None):
    """Write a small single-band Float64 .tif, like an Earth Engine S1 download."""
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(geotransform or [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype="float64"))
    dataset = None
    return path


def s1_filename(polar, suffix=""):
    return f"{DATE}_S1_{SITENAME}_{polar}{suffix}.tif"


def make_session(root, polarizations, suffix=""):
    """Create <root>/<site>/S1/<POLAR>/ with one .tif each, and return the inputs dict."""
    for index, polar in enumerate(polarizations):
        write_tif(root / SITENAME / "S1" / polar / s1_filename(polar, suffix), value=index)
    return {"filepath": str(root), "sitename": SITENAME}


# ------------------------------------------------------------------ create_folder_structure


def test_creates_one_folder_per_polarization(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", ["VV", "VH"])

    # the composite is built in memory, so it gets no folder of its own
    assert [Path(fp).name for fp in filepaths] == ["meta", "VV", "VH"]
    assert all(Path(fp).is_dir() for fp in filepaths)


def test_folders_are_created_in_canonical_order(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", ["VH", "VV"])

    assert [Path(fp).name for fp in filepaths] == ["meta", "VV", "VH"]


def test_defaults_to_vv_and_vh_when_no_polarizations_given(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1")

    assert [Path(fp).name for fp in filepaths] == ["meta", "VV", "VH"]


def test_accepts_a_bare_string_polarization(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", "VV")

    assert [Path(fp).name for fp in filepaths] == ["meta", "VV"]


def test_optical_layout_is_unchanged(tmp_path):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S2")

    assert [Path(fp).name for fp in filepaths] == ["meta", "ms", "swir", "mask"]


# ------------------------------------------------------------------ get_filepath


def test_get_filepath_returns_existing_polarization_folders(tmp_path):
    inputs = make_session(tmp_path, ["VV", "VH"])

    filepath = SDS_tools.get_filepath(inputs, "S1")

    assert [Path(fp).name for fp in filepath] == ["VV", "VH"]


def test_get_filepath_returns_canonical_order(tmp_path):
    # created VH first, but VV must still come first
    inputs = make_session(tmp_path, ["VH", "VV"])

    filepath = SDS_tools.get_filepath(inputs, "S1")

    assert [Path(fp).name for fp in filepath] == ["VV", "VH"]


def test_get_filepath_handles_a_legacy_vh_only_session(tmp_path):
    inputs = make_session(tmp_path, ["VH"])

    filepath = SDS_tools.get_filepath(inputs, "S1")

    assert [Path(fp).name for fp in filepath] == ["VH"]


def test_get_filepath_ignores_folders_that_do_not_exist(tmp_path):
    inputs = make_session(tmp_path, ["VV"])

    filepath = SDS_tools.get_filepath(inputs, "S1")

    assert [Path(fp).name for fp in filepath] == ["VV"]


# ------------------------------------------------------------------ get_filenames


def test_get_filenames_pairs_vv_and_vh(tmp_path):
    inputs = make_session(tmp_path, ["VV", "VH"])
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VV"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VV"), s1_filename("VH")]
    assert all(Path(p).exists() for p in fn)


def test_get_filenames_pairs_from_either_polarization(tmp_path):
    # the metadata records the first polarization, but starting from VH must work too
    inputs = make_session(tmp_path, ["VV", "VH"])
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VH"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VV"), s1_filename("VH")]


def test_get_filenames_pairs_duplicate_named_files(tmp_path):
    inputs = make_session(tmp_path, ["VV", "VH"], suffix="_dup0")
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VV", "_dup0"), filepath, "S1")

    assert [Path(p).name for p in fn] == [
        s1_filename("VV", "_dup0"),
        s1_filename("VH", "_dup0"),
    ]


def test_get_filenames_skips_a_polarization_missing_from_disk(tmp_path):
    # a scene downloaded before multi-polarization support, sitting in a VV+VH session
    inputs = make_session(tmp_path, ["VV", "VH"])
    (tmp_path / SITENAME / "S1" / "VV" / s1_filename("VV")).unlink()
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VH"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VH")]


def test_get_filenames_ignores_an_empty_polarization_folder(tmp_path):
    inputs = make_session(tmp_path, ["VH"])
    (tmp_path / SITENAME / "S1" / "HH").mkdir()
    filepath = SDS_tools.get_filepath(inputs, "S1")

    fn = SDS_tools.get_filenames(s1_filename("VH"), filepath, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VH")]


def test_get_filenames_raises_when_nothing_exists(tmp_path):
    inputs = make_session(tmp_path, ["VH"])
    filepath = SDS_tools.get_filepath(inputs, "S1")

    with pytest.raises(FileNotFoundError):
        SDS_tools.get_filenames(f"1999-01-01-00-00-00_S1_{SITENAME}_VH.tif", filepath, "S1")


def test_get_filenames_accepts_a_string_filepath(tmp_path):
    # get_filepath is monkeypatched to a plain string in test_extract_shorelines_all_zero_ms
    make_session(tmp_path, ["VH"])
    folder = str(tmp_path / SITENAME / "S1" / "VH")

    fn = SDS_tools.get_filenames(s1_filename("VH"), folder, "S1")

    assert [Path(p).name for p in fn] == [s1_filename("VH")]
