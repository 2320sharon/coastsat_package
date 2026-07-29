"""Tests for the SAR jpg preview: a titled false-colour composite of VV, VH and VV-VH."""

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_tools
from coastsat.SDS_preprocess import build_sar_preview, save_sar_image

DATE = "2023-12-03-19-15-49"
SIZE = 12


def gradient(low, high):
    """A band with a real spread, so the percentile stretch has something to work on."""
    return np.linspace(low, high, SIZE * SIZE).reshape(SIZE, SIZE).astype("float64")


def composite():
    """A [VV, VH, VV-VH] composite with three genuinely different dB ranges."""
    vv = gradient(-25.0, -2.0)
    vh = gradient(-32.0, -12.0)
    return np.dstack([vv, vh, vv - vh])


# ------------------------------------------------------------------ the displayed array


def test_composite_is_shown_as_rgb():
    im_display, _ = build_sar_preview(composite(), DATE, "S1")

    assert im_display.shape == (SIZE, SIZE, 3)


def test_each_channel_is_stretched_independently():
    # VV-VH spans a completely different dB range; a shared stretch would flatten it
    im_display, _ = build_sar_preview(composite(), DATE, "S1")

    for channel in range(3):
        band = im_display[:, :, channel]
        assert band.min() == pytest.approx(0.0, abs=1e-6)
        assert band.max() == pytest.approx(1.0, abs=1e-6)


def test_channels_keep_the_composite_band_order():
    im_composite = composite()
    im_display, _ = build_sar_preview(im_composite, DATE, "S1")

    for channel in range(3):
        # the stretch is monotonic, so the ordering of pixels must be preserved
        assert np.array_equal(
            np.argsort(im_display[:, :, channel], axis=None),
            np.argsort(im_composite[:, :, channel], axis=None),
        )


def test_display_values_are_within_the_unit_range():
    im_display, _ = build_sar_preview(composite(), DATE, "S1")

    assert im_display.min() >= 0.0
    assert im_display.max() <= 1.0


def test_single_polarization_is_shown_as_greyscale():
    im_display, _ = build_sar_preview(gradient(-32.0, -12.0)[:, :, None], DATE, "S1")

    assert im_display.ndim == 2


def test_two_dimensional_input_is_shown_as_greyscale():
    im_display, _ = build_sar_preview(gradient(-32.0, -12.0), DATE, "S1")

    assert im_display.ndim == 2


def test_flat_band_does_not_divide_by_zero():
    flat = np.full((SIZE, SIZE), -20.0)

    im_display, _ = build_sar_preview(np.dstack([flat, flat, flat - flat]), DATE, "S1")

    assert np.all(np.isfinite(im_display))


def test_nan_pixels_do_not_propagate():
    im_composite = composite()
    im_composite[0, 0, :] = np.nan

    im_display, _ = build_sar_preview(im_composite, DATE, "S1")

    assert np.all(np.isfinite(im_display))


# ------------------------------------------------------------------ the title


def test_title_names_the_composite_bands():
    _, title = build_sar_preview(composite(), DATE, "S1")

    assert "composite" in title.lower()
    for band in SDS_tools.SAR_COMPOSITE_BANDS:
        assert band in title


def test_title_maps_each_band_to_its_channel():
    _, title = build_sar_preview(composite(), DATE, "S1")

    assert "R=VV" in title
    assert "G=VH" in title
    assert "B=VV-VH" in title


def test_title_carries_the_date_and_satellite():
    _, title = build_sar_preview(composite(), DATE, "S1")

    assert DATE in title
    assert "S1" in title


def test_single_polarization_title_makes_no_composite_claim():
    _, title = build_sar_preview(gradient(-32.0, -12.0), DATE, "S1")

    assert "composite" not in title.lower()
    assert DATE in title


# ------------------------------------------------------------------ the saved file


def test_saves_a_jpg_into_the_rgb_folder(tmp_path):
    save_sar_image(composite(), DATE, "S1", str(tmp_path))

    assert (tmp_path / "RGB" / f"{DATE}_RGB_S1.jpg").exists()


def test_saves_a_jpg_for_a_single_polarization(tmp_path):
    save_sar_image(gradient(-32.0, -12.0)[:, :, None], DATE, "S1", str(tmp_path))

    assert (tmp_path / "RGB" / f"{DATE}_RGB_S1.jpg").exists()


def test_rejects_an_unexpected_shape(tmp_path):
    with pytest.raises(ValueError):
        save_sar_image(np.zeros((2, 2, 2, 2)), DATE, "S1", str(tmp_path))
