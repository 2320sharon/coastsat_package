from datetime import datetime
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_preprocess, SDS_shoreline


def test_extract_shorelines_skips_all_zero_s2_image_and_continues(
    monkeypatch, tmp_path
):
    logger = MagicMock()
    saved_outputs = []

    filenames = [
        "2024-01-01-00-00-00_S2_mock.tif",
        "2024-01-02-00-00-00_S2_mock.tif",
    ]

    settings = {
        "inputs": {
            "sitename": "unit-test-site",
            "filepath": str(tmp_path),
            "landsat_collection": "C02",
        },
        "cloud_thresh": 0.5,
        "dist_clouds": 300,
        "output_epsg": 32633,
        "check_detection": False,
        "adjust_detection": False,
        "save_figure": False,
        "min_beach_area": 25,
        "min_length_sl": 10,
        "cloud_mask_issue": False,
        "sand_color": "default",
        "pan_off": False,
    }

    metadata = {
        "S2": {
            "filenames": filenames,
            "dates": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "epsg": [32633, 32633],
            "acc_georef": [5.0, 5.0],
        }
    }

    monkeypatch.setattr(SDS_shoreline, "setup_logger", lambda *args, **kwargs: logger)
    monkeypatch.setattr(SDS_shoreline, "release_logger", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "get_filepath",
        lambda inputs, satname: str(tmp_path / satname),
    )
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "get_filenames",
        lambda filename, filepath, satname: [
            os.path.join(filepath, filename),
            os.path.join(filepath, f"swir_{filename}"),
            os.path.join(filepath, f"mask_{filename}"),
        ],
    )
    monkeypatch.setattr(SDS_shoreline, "load_classifier", lambda **kwargs: object())
    monkeypatch.setattr(
        SDS_shoreline,
        "create_shoreline_buffer",
        lambda im_shape, georef, image_epsg, pixel_size, settings: np.ones(
            im_shape, dtype=bool
        ),
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "classify_image_NN",
        lambda im_ms, cloud_mask, min_beach_area_pixels, clf: (
            np.zeros(cloud_mask.shape, dtype=int),
            np.zeros((*cloud_mask.shape, 3), dtype=bool),
        ),
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "find_shoreline_optical",
        lambda *args, **kwargs: (np.array([[0.0, 0.0], [1.0, 1.0]]), 0.12),
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "filter_shoreline",
        lambda shoreline, shoreline_extraction_area, output_epsg: shoreline,
    )
    monkeypatch.setattr(
        SDS_shoreline,
        "get_extract_shoreline_extraction_area_array",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        SDS_preprocess,
        "create_gdf_from_image_extent",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        SDS_preprocess,
        "write_to_json",
        lambda path, output: saved_outputs.append((path, output)),
    )

    def fake_preprocess_image(fn, satname, settings, collection):
        if os.path.basename(fn[0]).startswith("2024-01-01-00-00-00"):
            raise SDS_preprocess.SkipImageError(
                "Skipped image because Sentinel-2 multispectral file contains only zeros"
            )

        im_ms = np.ones((2, 2, 5), dtype=float)
        georef = np.array([0.0, 10.0, 0.0, 0.0, 0.0, -10.0])
        cloud_mask = np.zeros((2, 2), dtype=bool)
        im_nodata = np.zeros((2, 2), dtype=bool)
        return im_ms, georef, cloud_mask, im_nodata

    monkeypatch.setattr(SDS_shoreline, "preprocess_image", fake_preprocess_image)

    output = SDS_shoreline.extract_shorelines(metadata, settings)

    assert output["filename"] == [filenames[1]]
    assert output["dates"] == [metadata["S2"]["dates"][1]]
    assert output["idx"] == [1]
    assert output["satname"] == ["S2"]
    assert len(output["shorelines"]) == 1
    np.testing.assert_array_equal(
        output["shorelines"][0], np.array([[0.0, 0.0], [1.0, 1.0]])
    )
    assert saved_outputs and saved_outputs[0][1] == output
    assert any(
        "Skipped during preprocessing" in call.args[0]
        for call in logger.warning.call_args_list
    )
