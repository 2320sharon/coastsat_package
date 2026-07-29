"""Tests that a Sentinel-1 scene downloads every requested polarization.

Earth Engine is never contacted: download_S1_image is replaced with a stub that
writes a small .tif, so this exercises the folder layout, naming, metadata and the
rollback that stops a half-downloaded scene from being left on disk.
"""

import logging
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat import SDS_download, SDS_tools
from coastsat.SDS_download import process_sentinel1_image, read_metadata_file

DATE = "2023-12-03-19-15-49"
SITENAME = "site"
IMAGE_ID = "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20231203T191549"


def band(polarization):
    return {
        "id": polarization,
        "crs": "EPSG:32756",
        "crs_transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0],
        "dimensions": [25000, 16000],
    }


def im_meta(polarizations=("VV", "VH")):
    return {
        "id": IMAGE_ID,
        "bands": [band(polar) for polar in polarizations],
        "properties": {
            "orbitProperties_pass": "DESCENDING",
            "transmitterReceiverPolarisation": list(polarizations),
            "resolution": "H",
            "resolution_meters": 10,
            "instrumentMode": "IW",
        },
    }


def inputs_for(polarizations=("VV", "VH")):
    return {
        "sitename": SITENAME,
        "polygon": [[[151.3, -33.7], [151.4, -33.7], [151.4, -33.8], [151.3, -33.8]]],
        "sentinel_1_properties": {
            "transmitterReceiverPolarisation": list(polarizations),
            "instrumentMode": "IW",
        },
    }


# a distinct constant per polarization, so the composite bands are identifiable
VALUES = {"VV": -12.0, "VH": -20.0, "HH": -30.0}


def write_tif(path, value=-20.0, size=8):
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform([500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    dataset.GetRasterBand(1).WriteArray(np.full((size, size), value, dtype="float64"))
    dataset = None
    return str(path)


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the Earth Engine download with a stub that writes 'image.<POLAR>.tif'."""
    calls = []

    def _download(image_ee, polygon, polarization, save_path, **kwargs):
        calls.append(polarization)
        return write_tif(
            Path(save_path) / f"image.{polarization}.tif",
            VALUES.get(polarization, -20.0),
        )

    monkeypatch.setattr(SDS_download, "download_S1_image", _download)
    return calls


@pytest.fixture
def failing_download(monkeypatch):
    """Stub that succeeds for VV then fails for VH, to exercise the rollback."""

    def _download(image_ee, polygon, polarization, save_path, **kwargs):
        if polarization == "VH":
            raise RuntimeError("Earth Engine request failed")
        return write_tif(
            Path(save_path) / f"image.{polarization}.tif",
            VALUES.get(polarization, -20.0),
        )

    monkeypatch.setattr(SDS_download, "download_S1_image", _download)


@pytest.fixture
def logger():
    return logging.getLogger("test_s1_download")


def run(tmp_path, polarizations, meta, logger, all_names=None):
    filepaths = SDS_tools.create_folder_structure(
        str(tmp_path), "S1", list(polarizations)
    )
    result = process_sentinel1_image(
        image_ee=None,
        im_meta=meta,
        inputs=inputs_for(polarizations),
        filepaths=filepaths,
        im_date=DATE,
        suffix=".tif",
        all_names=[] if all_names is None else all_names,
        logger=logger,
    )
    return result, filepaths


def s1_dir(tmp_path, polar):
    return tmp_path / "S1" / polar


# ------------------------------------------------------------------ happy path


def test_downloads_every_requested_polarization(tmp_path, fake_download, logger):
    result, _ = run(tmp_path, ["VV", "VH"], im_meta(), logger)

    assert result["skip_image"] is False
    assert fake_download == ["VV", "VH"]
    assert (s1_dir(tmp_path, "VV") / f"{DATE}_S1_{SITENAME}_VV.tif").exists()
    assert (s1_dir(tmp_path, "VH") / f"{DATE}_S1_{SITENAME}_VH.tif").exists()


def test_downloads_in_canonical_order(tmp_path, fake_download, logger):
    run(tmp_path, ["VH", "VV"], im_meta(), logger)

    assert fake_download == ["VV", "VH"]


def test_reports_the_band_order(tmp_path, fake_download, logger):
    result, _ = run(tmp_path, ["VV", "VH"], im_meta(), logger)

    assert result["band_order"] == ["VV", "VH"]
    assert result["filename_ms"] == f"{DATE}_S1_{SITENAME}_VV.tif"


def test_writes_one_metadata_file_for_the_scene(tmp_path, fake_download, logger):
    _, filepaths = run(tmp_path, ["VV", "VH"], im_meta(), logger)

    meta_files = sorted(Path(filepaths[0]).glob("*.txt"))
    assert [f.name for f in meta_files] == [f"{DATE}_S1_{SITENAME}.txt"]


def test_metadata_records_the_band_order(tmp_path, fake_download, logger):
    _, filepaths = run(tmp_path, ["VV", "VH"], im_meta(), logger)

    meta_info = read_metadata_file(str(Path(filepaths[0]) / f"{DATE}_S1_{SITENAME}.txt"))

    assert meta_info["band_order"] == ["VV", "VH"]
    assert meta_info["filename"] == f"{DATE}_S1_{SITENAME}_VV.tif"
    assert meta_info["epsg"] == 32756
    assert meta_info["instrumentMode"] == "IW"


def test_single_polarization_still_works(tmp_path, fake_download, logger):
    result, _ = run(tmp_path, ["VH"], im_meta(), logger)

    assert result["band_order"] == ["VH"]
    assert fake_download == ["VH"]
    assert (s1_dir(tmp_path, "VH") / f"{DATE}_S1_{SITENAME}_VH.tif").exists()


def test_each_polarization_is_registered_once(tmp_path, fake_download, logger):
    # get_file_name() adds the name itself; adding it again would misnumber duplicates
    all_names = []
    run(tmp_path, ["VV", "VH"], im_meta(), logger, all_names=all_names)

    assert all_names == [
        f"{DATE}_S1_{SITENAME}_VV.tif",
        f"{DATE}_S1_{SITENAME}_VH.tif",
    ]


def test_duplicate_scene_gets_matching_dup_suffixes(tmp_path, fake_download, logger):
    all_names = []
    run(tmp_path, ["VV", "VH"], im_meta(), logger, all_names=all_names)
    result, _ = run(tmp_path, ["VV", "VH"], im_meta(), logger, all_names=all_names)

    # both polarizations of the second scene must carry the SAME _dup counter,
    # or the read path could not pair them back up
    assert result["filename_ms"] == f"{DATE}_S1_{SITENAME}_VV_dup0.tif"
    assert (s1_dir(tmp_path, "VV") / f"{DATE}_S1_{SITENAME}_VV_dup0.tif").exists()
    assert (s1_dir(tmp_path, "VH") / f"{DATE}_S1_{SITENAME}_VH_dup0.tif").exists()


def test_only_the_polarization_folders_are_written(tmp_path, fake_download, logger):
    # the composite is built in memory for the preview, never saved as a .tif
    run(tmp_path, ["VV", "VH"], im_meta(), logger)

    assert sorted(p.name for p in (tmp_path / "S1").iterdir()) == ["VH", "VV", "meta"]


# ------------------------------------------------------------------ skipping and rollback


def test_skips_a_scene_missing_a_requested_polarization(tmp_path, fake_download, logger):
    result, filepaths = run(tmp_path, ["VV", "VH"], im_meta(["VV"]), logger)

    assert result["skip_image"] is True
    assert fake_download == []  # nothing downloaded at all
    assert list(Path(filepaths[0]).glob("*.txt")) == []


def test_partial_download_is_rolled_back(tmp_path, failing_download, logger):
    result, filepaths = run(tmp_path, ["VV", "VH"], im_meta(), logger)

    assert result["skip_image"] is True
    # the VV file that did download must not be left behind
    assert list(s1_dir(tmp_path, "VV").glob("*.tif")) == []
    assert list(s1_dir(tmp_path, "VH").glob("*.tif")) == []
    assert list(Path(filepaths[0]).glob("*.txt")) == []


def test_invalid_polarization_skips_the_image(tmp_path, fake_download, logger):
    filepaths = SDS_tools.create_folder_structure(str(tmp_path), "S1", ["VH"])
    inputs = inputs_for(["VH"])
    inputs["sentinel_1_properties"]["transmitterReceiverPolarisation"] = ["XX"]

    result = process_sentinel1_image(
        image_ee=None,
        im_meta=im_meta(),
        inputs=inputs,
        filepaths=filepaths,
        im_date=DATE,
        suffix=".tif",
        all_names=[],
        logger=logger,
    )

    assert result["skip_image"] is True
