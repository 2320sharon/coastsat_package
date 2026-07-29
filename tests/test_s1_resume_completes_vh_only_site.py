"""End-to-end resume of a legacy VH-only Sentinel-1 site.

test_s1_resume_download.py covers the helpers in isolation, against hand-built metadata
dicts. This drives the whole resume instead: a real VH-only site on disk (VH .tif plus a
'_VH.txt' metadata file carrying 'saved_polarization', exactly as written before
multi-polarization support), read back by the real get_metadata, filtered by
remove_existing_imagery, then downloaded by process_sentinel1_image with Earth Engine
stubbed out.

The point of the fix is not that a scene is *queued* - it is that VV ends up on disk and
the scene can finally build the 3-channel [VV, VH, VV-VH] composite the segmentation
model needs. That is what these tests assert.
"""

import logging
from pathlib import Path
import sys

import numpy as np
import pytest
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from osgeo import gdal

from coastsat import SDS_download, SDS_tools
from coastsat.SDS_download import (
    get_metadata,
    process_sentinel1_image,
    remove_existing_imagery,
    write_metadata_file,
)

SITENAME = "site"
STAMP = "2023-12-03-19-15-49"
OTHER_STAMP = "2023-12-06-08-39-48"
IMAGE_ID = "COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20231203T191549"
GEOTRANSFORM = [500000.0, 10.0, 0.0, 6000000.0, 0.0, -10.0]
SIZE = 8

# a distinct constant per polarization, so a band can be identified after stacking
VALUES = {"VV": -12.0, "VH": -20.0}


def write_tif(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), SIZE, SIZE, 1, gdal.GDT_Float64)
    dataset.SetGeoTransform(GEOTRANSFORM)
    dataset.GetRasterBand(1).WriteArray(np.full((SIZE, SIZE), value, dtype="float64"))
    dataset = None
    return str(path)


def legacy_metadict(stamp):
    """A metadata file as written before multi-polarization support: no 'band_order'."""
    return {
        "filename": f"{stamp}_S1_{SITENAME}_VH.tif",
        "epsg": 32756,
        "im_width": SIZE,
        "im_height": SIZE,
        "orbitProperties_pass": "DESCENDING",
        "transmitterReceiverPolarisation": ["VV", "VH"],
        "saved_polarization": "VH",
        "resolution": "H",
        "resolution_meters": 10,
        "instrumentMode": "IW",
    }


def build_legacy_vh_site(tmp_path, stamps=(STAMP,)):
    """Create a site as an older coastsat left it: S1/VH/ and '*_VH.txt' only."""
    site = tmp_path / SITENAME
    meta_dir = site / "S1" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        write_tif(site / "S1" / "VH" / f"{stamp}_S1_{SITENAME}_VH.tif", VALUES["VH"])
        write_metadata_file(
            str(meta_dir), f"{stamp}_S1_{SITENAME}", legacy_metadict(stamp), polar="VH"
        )
    return site


def inputs_for(tmp_path, polarizations=("VV", "VH")):
    return {
        "filepath": str(tmp_path),
        "sitename": SITENAME,
        "sat_list": ["S1"],
        "dates": ["2020-01-01", "2030-01-01"],
        "polygon": [[[151.3, -33.7], [151.4, -33.7], [151.4, -33.8], [151.3, -33.8]]],
        "sentinel_1_properties": {
            "transmitterReceiverPolarisation": list(polarizations),
            "instrumentMode": "IW",
        },
    }


def band(polarization):
    return {
        "id": polarization,
        "crs": "EPSG:32756",
        "crs_transform": [10.0, 0.0, 500000.0, 0.0, -10.0, 6000000.0],
        "dimensions": [SIZE, SIZE],
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
            "system:time_start": 0,  # replaced per scene by image_dict_for
        },
    }


def image_dict_for(stamps):
    """The 'available on the server' dict, as retrieve_images builds it from GEE."""
    images = []
    for stamp in stamps:
        image = im_meta()
        date = SDS_download.parse_date_from_filename(stamp)
        image["properties"]["system:time_start"] = date.timestamp() * 1000
        images.append(image)
    return {"S1": images}


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the Earth Engine download with a stub that writes 'image.<POLAR>.tif'."""
    calls = []

    def _download(image_ee, polygon, polarization, save_path, **kwargs):
        calls.append(polarization)
        return write_tif(
            Path(save_path) / f"image.{polarization}.tif", VALUES[polarization]
        )

    monkeypatch.setattr(SDS_download, "download_S1_image", _download)
    return calls


def resume(tmp_path, inputs):
    """Run the resume the way retrieve_images does: filter, then download what is left.

    Returns the image_dict that survived remove_existing_imagery.
    """
    logger = logging.getLogger("test_s1_resume_completes_vh_only_site")
    polarizations = SDS_download.get_sentinel1_polarizations(inputs)
    stamps = [
        SDS_download.parse_date_from_filename(name).strftime("%Y-%m-%d-%H-%M-%S")
        for name in [STAMP, OTHER_STAMP]
    ]
    available = image_dict_for(
        [stamp for stamp in stamps if _scene_exists(tmp_path, stamp)]
    )

    to_download = remove_existing_imagery(
        available, get_metadata(inputs), ["S1"], inputs
    )

    filepaths = SDS_tools.create_folder_structure(
        str(tmp_path / SITENAME), "S1", list(polarizations)
    )
    all_names = []
    for image in to_download["S1"]:
        im_date = SDS_download.parse_date_from_filename(
            _stamp_of(image)
        ).strftime("%Y-%m-%d-%H-%M-%S")
        process_sentinel1_image(
            image_ee=None,
            im_meta=image,
            inputs=inputs,
            filepaths=filepaths,
            im_date=im_date,
            suffix=".tif",
            all_names=all_names,
            logger=logger,
        )
    return to_download


def _stamp_of(image):
    from datetime import datetime

    return datetime.fromtimestamp(
        image["properties"]["system:time_start"] / 1000, tz=pytz.utc
    ).strftime("%Y-%m-%d-%H-%M-%S")


def _scene_exists(tmp_path, stamp):
    """Whether the server offers this scene - here, whether the site was built with it."""
    meta_dir = tmp_path / SITENAME / "S1" / "meta"
    return any(meta_dir.glob(f"{stamp}_S1_{SITENAME}*.txt"))


def tif(tmp_path, polar, stamp=STAMP):
    return tmp_path / SITENAME / "S1" / polar / f"{stamp}_S1_{SITENAME}_{polar}.tif"


# ------------------------------------------------------------------ the precondition


def test_the_legacy_site_starts_out_vh_only(tmp_path):
    build_legacy_vh_site(tmp_path)

    metadata = get_metadata(inputs_for(tmp_path))

    assert metadata["S1"]["band_order"] == [["VH"]]  # via 'saved_polarization'
    assert not (tmp_path / SITENAME / "S1" / "VV").exists()


def test_the_legacy_scene_cannot_build_the_model_composite(tmp_path):
    # this is what the resume has to fix: no VV means no 3-channel composite, so the
    # scene falls back to Otsu and can never reach the segmentation model
    build_legacy_vh_site(tmp_path)
    inputs = inputs_for(tmp_path)
    metadata = get_metadata(inputs)

    im_polar, _ = SDS_tools.read_sar_image(
        SDS_tools.get_filenames(
            metadata["S1"]["filenames"][0], SDS_tools.get_filepath(inputs, "S1"), "S1"
        )
    )

    assert im_polar.shape == (SIZE, SIZE, 1)
    assert SDS_tools.build_sar_composite(im_polar, ["VH"]) is None


# ------------------------------------------------------------------ the resume


def test_resume_queues_the_vh_only_scene(tmp_path, fake_download):
    build_legacy_vh_site(tmp_path)

    to_download = resume(tmp_path, inputs_for(tmp_path))

    assert len(to_download["S1"]) == 1, "the incomplete scene was not queued"


def test_resume_writes_the_missing_vv_to_disk(tmp_path, fake_download):
    build_legacy_vh_site(tmp_path)

    resume(tmp_path, inputs_for(tmp_path))

    assert tif(tmp_path, "VV").exists(), "VV was never downloaded"
    assert fake_download == ["VV", "VH"]


def test_resume_keeps_exactly_one_vh(tmp_path, fake_download):
    # the scene is re-downloaded whole, so VH is fetched again - it must overwrite the
    # existing file rather than land beside it as a '_dup0'
    build_legacy_vh_site(tmp_path)

    resume(tmp_path, inputs_for(tmp_path))

    vh_files = sorted(p.name for p in (tmp_path / SITENAME / "S1" / "VH").glob("*.tif"))
    assert vh_files == [f"{STAMP}_S1_{SITENAME}_VH.tif"]


def test_after_the_resume_the_scene_reports_both_polarizations(tmp_path, fake_download):
    build_legacy_vh_site(tmp_path)
    inputs = inputs_for(tmp_path)

    resume(tmp_path, inputs)

    band_orders = get_metadata(inputs)["S1"]["band_order"]
    assert ["VV", "VH"] in band_orders


def test_after_the_resume_the_scene_builds_the_model_composite(tmp_path, fake_download):
    # the whole point: the completed scene now stacks to the 3-channel composite the
    # ONNX model needs, instead of falling back to Otsu on VH
    build_legacy_vh_site(tmp_path)
    inputs = inputs_for(tmp_path)
    resume(tmp_path, inputs)

    filepath = SDS_tools.get_filepath(inputs, "S1")
    file_paths = SDS_tools.get_filenames(
        f"{STAMP}_S1_{SITENAME}_VV.tif", filepath, "S1"
    )
    im_polar, _ = SDS_tools.read_sar_image(file_paths)
    composite = SDS_tools.build_sar_composite(im_polar, ["VV", "VH"])

    assert [Path(fp).parent.name for fp in file_paths] == ["VV", "VH"]
    assert im_polar.shape == (SIZE, SIZE, 2)
    assert composite is not None
    assert composite.shape == (SIZE, SIZE, 3)
    # channels in canonical order: VV, VH, VV-VH
    assert np.allclose(composite[:, :, 0], VALUES["VV"])
    assert np.allclose(composite[:, :, 1], VALUES["VH"])
    assert np.allclose(composite[:, :, 2], VALUES["VV"] - VALUES["VH"])


def test_a_second_resume_downloads_nothing(tmp_path, fake_download):
    # a completed site must not re-download on every run
    build_legacy_vh_site(tmp_path)
    inputs = inputs_for(tmp_path)
    resume(tmp_path, inputs)
    fake_download.clear()

    to_download = resume(tmp_path, inputs)

    assert to_download["S1"] == []
    assert fake_download == []


def test_a_vh_only_request_leaves_the_legacy_site_alone(tmp_path, fake_download):
    # a user who pins transmitterReceiverPolarisation to VH must not have their whole
    # site re-downloaded by the new completeness check
    build_legacy_vh_site(tmp_path)

    to_download = resume(tmp_path, inputs_for(tmp_path, ["VH"]))

    assert to_download["S1"] == []
    assert fake_download == []
    assert not (tmp_path / SITENAME / "S1" / "VV").exists()


def test_the_stale_legacy_metadata_file_is_left_behind(tmp_path, fake_download):
    """Documents current behaviour, not desired behaviour.

    The legacy file is '<stamp>_S1_<site>_VH.txt'; the resume writes the modern
    '<stamp>_S1_<site>.txt' beside it and never removes the old one, so get_metadata
    lists the completed scene twice - once as ['VV','VH'] and once as ['VH'].

    The resume itself stays correct: get_downloaded_sar_polarizations unions the
    polarizations per date, so both entries resolve to VV+VH and a second run downloads
    nothing (see test_a_second_resume_downloads_nothing). The cost is that extraction
    maps the scene twice, producing duplicate shorelines for that date unless
    SDS_tools.remove_duplicates is called.
    """
    build_legacy_vh_site(tmp_path)
    inputs = inputs_for(tmp_path)

    resume(tmp_path, inputs)

    meta_dir = tmp_path / SITENAME / "S1" / "meta"
    assert sorted(p.name for p in meta_dir.iterdir()) == [
        f"{STAMP}_S1_{SITENAME}.txt",
        f"{STAMP}_S1_{SITENAME}_VH.txt",
    ]
    metadata = get_metadata(inputs)["S1"]
    assert len(metadata["dates"]) == 2
    assert metadata["band_order"] == [["VV", "VH"], ["VH"]]
    # both entries point at the same scene, so both read the same two files
    for filename in metadata["filenames"]:
        paths = SDS_tools.get_filenames(
            filename, SDS_tools.get_filepath(inputs, "S1"), "S1"
        )
        assert [Path(p).parent.name for p in paths] == ["VV", "VH"]


def test_resume_completes_only_the_incomplete_scene(tmp_path, fake_download):
    # a site holding one dual-pol scene and one legacy VH-only scene: only the legacy
    # one is fetched again
    build_legacy_vh_site(tmp_path, stamps=(STAMP, OTHER_STAMP))
    inputs = inputs_for(tmp_path)
    # promote the first scene to VV+VH, as a newer download would have left it
    write_tif(tif(tmp_path, "VV"), VALUES["VV"])
    write_metadata_file(
        str(tmp_path / SITENAME / "S1" / "meta"),
        f"{STAMP}_S1_{SITENAME}",
        {**legacy_metadict(STAMP), "band_order": ["VV", "VH"]},
    )
    (tmp_path / SITENAME / "S1" / "meta" / f"{STAMP}_S1_{SITENAME}_VH.txt").unlink()

    to_download = resume(tmp_path, inputs)

    assert len(to_download["S1"]) == 1
    assert _stamp_of(to_download["S1"][0]) == OTHER_STAMP
    assert tif(tmp_path, "VV", OTHER_STAMP).exists()
