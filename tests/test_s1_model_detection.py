"""Tests that S1 scenes are segmented with the ONNX model, falling back to Otsu.

The model is the default, but it needs onnxruntime, the .onnx file and a dual-polarization
scene. Whenever any of those is missing the run must still map a shoreline with the legacy
Otsu threshold rather than dropping the image.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_sar_model, SDS_shoreline
from coastsat.SDS_sar_model import SarModelUnavailable
from coastsat.SDS_shoreline import (
    SAR_SEGMENTATION_DEFAULT,
    load_sar_segmenter,
    segment_sar_water,
)

DATE = "2023-12-03-19-15-49"
SIZE = 8


def s1_path(polar):
    return str(Path("data") / "site" / "S1" / polar / f"{DATE}_S1_site_{polar}.tif")


DUAL_POL = [s1_path("VV"), s1_path("VH")]


def composite(height=SIZE, width=SIZE):
    im_vv = np.full((height, width), -12.0)
    im_vh = np.full((height, width), -20.0)
    return np.stack([im_vv, im_vh, im_vv - im_vh], axis=2)


@pytest.fixture(autouse=True)
def clear_cache():
    SDS_sar_model.clear_model_cache()
    yield
    SDS_sar_model.clear_model_cache()


class RecordingLogger:
    def __init__(self):
        self.info_messages = []
        self.warnings = []

    def info(self, message):
        self.info_messages.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))


# --------------------------------------------------------------- load_sar_segmenter


def test_the_model_is_the_default_method():
    assert SAR_SEGMENTATION_DEFAULT == "model"


def test_loads_the_model_by_default(monkeypatch):
    loaded = []
    monkeypatch.setattr(
        SDS_sar_model,
        "load_sar_model",
        lambda path: loaded.append(path) or ("session", {"channel_order": ["VV", "VH", "VV-VH"]}),
    )

    segmenter = load_sar_segmenter({}, logger=RecordingLogger())

    assert segmenter is not None
    # None means "the default model", which authorizes load_sar_model to download it
    assert loaded == [None]


def test_sar_segmentation_otsu_skips_the_model_entirely(monkeypatch):
    monkeypatch.setattr(
        SDS_sar_model,
        "load_sar_model",
        lambda path: pytest.fail("the model must not be loaded when Otsu is requested"),
    )
    logger = RecordingLogger()

    assert load_sar_segmenter({"sar_segmentation": "otsu"}, logger=logger) is None
    assert any("otsu" in message.lower() for message in logger.info_messages)


def test_falls_back_when_onnxruntime_is_missing(monkeypatch):
    def unavailable(_path):
        raise SarModelUnavailable("onnxruntime is not installed")

    monkeypatch.setattr(SDS_sar_model, "load_sar_model", unavailable)
    logger = RecordingLogger()

    assert load_sar_segmenter({}, logger=logger) is None
    assert any("onnxruntime" in warning for warning in logger.warnings)


def test_falls_back_when_the_model_file_is_missing(tmp_path):
    logger = RecordingLogger()
    settings = {"sar_model_path": str(tmp_path / "missing.onnx")}

    assert load_sar_segmenter(settings, logger=logger) is None
    assert any("not found" in warning for warning in logger.warnings)


def test_override_paths_are_never_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(
        SDS_sar_model,
        "fetch_default_sar_model",
        lambda: pytest.fail("an explicit sar_model_path must never trigger a download"),
    )
    logger = RecordingLogger()
    settings = {"sar_model_path": str(tmp_path / "missing.onnx")}

    assert load_sar_segmenter(settings, logger=logger) is None
    assert any("not found" in warning for warning in logger.warnings)


def test_settings_can_point_at_another_model(monkeypatch, tmp_path):
    requested = []
    monkeypatch.setattr(
        SDS_sar_model,
        "load_sar_model",
        lambda path: requested.append(path) or ("session", {"channel_order": []}),
    )

    load_sar_segmenter({"sar_model_path": str(tmp_path / "other.onnx")})

    assert Path(requested[0]).name == "other.onnx"


# --------------------------------------------------------------- segment_sar_water


def test_returns_none_without_a_segmenter():
    assert segment_sar_water(composite(), DUAL_POL, None, {}) is None


def test_returns_the_water_validity_and_probability_masks(monkeypatch):
    valid = np.ones((SIZE, SIZE), dtype=bool)
    water = np.zeros((SIZE, SIZE), dtype=bool)
    water[4:, :] = True
    prob = np.where(water, 0.9, 0.1).astype(np.float32)
    monkeypatch.setattr(SDS_shoreline.SDS_tools, "read_sar_valid_mask", lambda fn: valid)
    monkeypatch.setattr(
        SDS_sar_model, "segment_water", lambda *args, **kwargs: (water, prob)
    )

    im_water, returned_valid, im_prob = segment_sar_water(
        composite(), DUAL_POL, ("session", {}), {}
    )

    assert np.array_equal(im_water, water)
    assert np.array_equal(returned_valid, valid)
    # the probability the mask was cut from, for save_sar_probability
    assert np.array_equal(im_prob, prob)


def test_falls_back_for_a_single_polarization_scene(monkeypatch):
    # a legacy VH-only scene is (H, W, 1), which the model cannot take
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "read_sar_valid_mask",
        lambda fn: np.ones((SIZE, SIZE), dtype=bool),
    )
    logger = RecordingLogger()
    im_ms = np.full((SIZE, SIZE, 1), -20.0)
    spec = SDS_sar_model.read_model_spec(_SpecOnlySession())

    result = segment_sar_water(
        im_ms, [s1_path("VH")], (None, spec), {}, logger=logger, date_str=DATE
    )

    assert result is None
    assert any("missing VV or VH" in warning for warning in logger.warnings)


def test_falls_back_when_inference_fails(monkeypatch):
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "read_sar_valid_mask",
        lambda fn: np.ones((SIZE, SIZE), dtype=bool),
    )

    def boom(*args, **kwargs):
        raise SarModelUnavailable("SAR model inference failed: out of memory")

    monkeypatch.setattr(SDS_sar_model, "segment_water", boom)
    logger = RecordingLogger()

    result = segment_sar_water(
        composite(), DUAL_POL, ("session", {}), {}, logger=logger, date_str=DATE
    )

    assert result is None
    assert any("out of memory" in warning for warning in logger.warnings)


def test_the_water_threshold_setting_is_passed_through(monkeypatch):
    monkeypatch.setattr(
        SDS_shoreline.SDS_tools,
        "read_sar_valid_mask",
        lambda fn: np.ones((SIZE, SIZE), dtype=bool),
    )
    thresholds = []
    monkeypatch.setattr(
        SDS_sar_model,
        "segment_water",
        lambda im, valid, session, spec, threshold: thresholds.append(threshold)
        or (np.zeros((SIZE, SIZE), dtype=bool), None),
    )

    segment_sar_water(composite(), DUAL_POL, ("session", {}), {})
    segment_sar_water(
        composite(), DUAL_POL, ("session", {}), {"sar_water_threshold": 0.8}
    )

    assert thresholds == [SDS_sar_model.WATER_THRESHOLD, 0.8]


class _SpecOnlySession:
    """Minimal stand-in used only to build a real spec dict.

    It has to carry the metadata a real export carries: read_model_spec rejects a model
    that does not declare the constants it was trained with.
    """

    class _Meta:
        custom_metadata_map = {
            "channel_order": '["VV", "VH", "VV-VH"]',
            "normalization_mean": "[-12.59, -20.26, 10.5465]",
            "normalization_std": "[5.26, 5.91, 7.6855]",
        }

    def get_modelmeta(self):
        return self._Meta()


# --------------------------------------------------------------- shoreline search area


def test_invalid_pixels_are_excluded_from_the_contour_search():
    # find_shoreline_SAR NaNs out everything outside the mask it is given, and
    # process_contours then drops the NaN points, so nodata cannot make a shoreline
    water = np.zeros((10, 10), dtype=bool)
    water[5:, :] = True
    buffer = np.ones((10, 10), dtype=bool)
    valid = np.ones((10, 10), dtype=bool)
    valid[:, 8:] = False
    settings = {"output_epsg": 4326, "min_length_sl": 0}
    georef = np.array([0.0, 1.0, 0.0, 0.0, 0.0, -1.0])

    with_all = SDS_shoreline.find_shoreline_SAR(water, buffer, georef, 4326, settings)
    with_valid = SDS_shoreline.find_shoreline_SAR(
        water, buffer & valid, georef, 4326, settings
    )

    assert len(with_valid) < len(with_all)
