"""End-to-end check of the real ONNX model on a real downloaded scene.

Skipped unless onnxruntime is installed, the 130 MB model is present, and a
dual-polarization session exists on disk. Everything else about the model path is
covered by the stubbed tests in test_sar_model.py.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_sar_model, SDS_tools

REPO_ROOT = Path(__file__).resolve().parents[1]
SESSION = REPO_ROOT / "data" / "test_s1_downloads_7_24" / "S1"

onnxruntime = pytest.importorskip("onnxruntime", reason="onnxruntime is not installed")


def scene_paths():
    """The VV/VH pair of the first scene of the session, or None."""
    if not (SESSION / "VV").is_dir() or not (SESSION / "VH").is_dir():
        return None
    for vv in sorted((SESSION / "VV").glob("*_VV.tif")):
        vh = SESSION / "VH" / vv.name.replace("_VV.tif", "_VH.tif")
        if vh.exists():
            return str(vv), str(vh)
    return None


PAIR = scene_paths()

pytestmark = [
    pytest.mark.skipif(
        not Path(SDS_sar_model.get_sar_model_path({})).is_file(),
        reason="the SAR model .onnx is not present",
    ),
    pytest.mark.skipif(PAIR is None, reason="no dual-polarization scene on disk"),
]


@pytest.fixture(scope="module")
def segmenter():
    return SDS_sar_model.load_sar_model()


@pytest.fixture(scope="module")
def scene():
    im_ms, _georef = SDS_tools.read_sar_image(list(PAIR))
    composite = SDS_tools.build_sar_composite(im_ms, ["VV", "VH"])
    valid = SDS_tools.read_sar_valid_mask(list(PAIR))
    return composite, valid


def test_the_packaged_model_declares_the_expected_contract(segmenter):
    _session, spec = segmenter

    assert list(spec["channel_order"]) == list(SDS_tools.SAR_COMPOSITE_BANDS)
    assert np.allclose(spec["mean"], [-12.59, -20.26, 10.5465])
    assert np.allclose(spec["std"], [5.26, 5.91, 7.6855])
    assert spec["output_stride"] == 32
    assert str(spec["input_scale"]).lower() == "db"


def test_the_downloaded_rasters_are_in_db(scene):
    composite, _valid = scene

    # sanity check from the model's guide: VV over land is roughly -5 to -15 dB and
    # VH lower still. A linear-power raster would be near 0 and predict one class.
    assert -30 < np.median(composite[:, :, 0]) < 0
    assert np.median(composite[:, :, 1]) < np.median(composite[:, :, 0])


def test_segments_a_real_scene_into_land_and_water(segmenter, scene):
    session, spec = segmenter
    composite, valid = scene

    im_water, prob = SDS_sar_model.segment_water(composite, valid, session, spec)

    assert im_water.shape == composite.shape[:2]
    assert prob.shape == composite.shape[:2]
    # a coastal scene must contain both classes; all-one-class means the input scale
    # or the channel order is wrong
    assert 0.05 < im_water.mean() < 0.95
    assert np.nanmin(prob) >= 0.0 and np.nanmax(prob) <= 1.0


def test_a_higher_threshold_never_grows_the_water_class(segmenter, scene):
    session, spec = segmenter
    composite, valid = scene

    lenient, _ = SDS_sar_model.segment_water(composite, valid, session, spec, 0.5)
    strict, _ = SDS_sar_model.segment_water(composite, valid, session, spec, 0.9)

    assert strict.sum() <= lenient.sum()
    assert np.all(lenient[strict])  # strict water is a subset of lenient water


def test_the_session_is_cached(segmenter):
    session, _spec = segmenter

    again, _ = SDS_sar_model.load_sar_model()

    assert again is session
