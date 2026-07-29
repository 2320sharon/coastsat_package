"""Tests that SAR jpg previews are written from the images on disk.

The S1 branch of save_jpg used to be handed a bare filename rather than a full path,
so it only worked when the current directory happened to be the .tif folder.
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat import SDS_preprocess

DATE = "2023-12-03-19-15-49"
SITENAME = "site"


def write_tif(path, value, size=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), size, size, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform([500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0])
    # a gradient, so rescale_intensity has a range to work with
    array = np.linspace(value, value + 15.0, size * size).reshape(size, size)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset = None


def make_session(root, polarizations):
    for polar in polarizations:
        write_tif(
            root / SITENAME / "S1" / polar / f"{DATE}_S1_{SITENAME}_{polar}.tif",
            value=-30.0 if polar == "VV" else -25.0,
        )
    metadata = {
        "S1": {"filenames": [f"{DATE}_S1_{SITENAME}_{polarizations[0]}.tif"]}
    }
    settings = {
        "inputs": {"filepath": str(root), "sitename": SITENAME, "landsat_collection": "C02"},
        "cloud_thresh": 0.5,
        "cloud_mask_issue": False,
        "pan_off": False,
    }
    return metadata, settings


def preview_path(root):
    return root / SITENAME / "jpg_files" / "preprocessed" / "RGB" / f"{DATE}_RGB_S1.jpg"


def test_writes_a_preview_for_a_dual_polarization_scene(tmp_path, monkeypatch):
    # run from somewhere else entirely: a bare filename would not resolve
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = make_session(tmp_path, ["VV", "VH"])

    SDS_preprocess.save_jpg(metadata, settings)

    assert preview_path(tmp_path).exists()


def test_writes_a_preview_for_a_legacy_vh_only_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = make_session(tmp_path, ["VH"])

    SDS_preprocess.save_jpg(metadata, settings)

    assert preview_path(tmp_path).exists()


def test_writes_a_preview_for_a_duplicate_named_scene(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = make_session(tmp_path, ["VV", "VH"])
    for polar in ("VV", "VH"):
        folder = tmp_path / SITENAME / "S1" / polar
        (folder / f"{DATE}_S1_{SITENAME}_{polar}.tif").rename(
            folder / f"{DATE}_S1_{SITENAME}_{polar}_dup0.tif"
        )
    metadata["S1"]["filenames"] = [f"{DATE}_S1_{SITENAME}_VV_dup0.tif"]

    SDS_preprocess.save_jpg(metadata, settings)

    assert preview_path(tmp_path).exists()


def test_no_composite_tif_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path.parent)
    metadata, settings = make_session(tmp_path, ["VV", "VH"])

    SDS_preprocess.save_jpg(metadata, settings)

    s1_dir = tmp_path / SITENAME / "S1"
    assert sorted(p.name for p in s1_dir.iterdir()) == ["VH", "VV"]
