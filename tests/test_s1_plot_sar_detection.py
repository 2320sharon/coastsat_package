"""Tests that the SAR detection figure handles any number of polarization channels.

matplotlib's imshow only accepts 1, 3 or 4 trailing channels, so a 2-channel VV+VH
array has to be reduced to one band before it is displayed.
"""

from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_plotting

DATE = "2023-12-03-19-15-49"


@pytest.fixture
def detection_args():
    height, width = 12, 10
    return {
        "im_labels": np.zeros((height, width), dtype=bool),
        "sl_pix": np.array([[1.0, 1.0], [2.0, 2.0]]),
        "date": DATE,
        "satname": "S1",
        "im_ref_buffer": np.ones((height, width), dtype=bool),
        "shoreline_extraction_area": [],
    }


def sar_image(n_channels, height=12, width=10):
    """A SAR image whose channels differ, so the displayed one is identifiable."""
    channels = [
        np.linspace(-30 - 10 * c, -5 - 10 * c, height * width).reshape(height, width)
        for c in range(n_channels)
    ]
    return np.stack(channels, axis=2)


@pytest.mark.parametrize("n_channels", [1, 2, 3, 4])
def test_handles_any_number_of_channels(tmp_path, detection_args, n_channels):
    # 2 channels is VV+VH, 1 is a legacy VH-only scene, 3 is the pre-change array shape
    SDS_plotting.plot_sar_detection(
        sar_image(n_channels),
        output_path=str(tmp_path),
        settings={"save_figure": True},
        **detection_args,
    )

    assert (tmp_path / f"{DATE}_S1.jpg").exists()


def test_handles_a_two_dimensional_image(tmp_path, detection_args):
    SDS_plotting.plot_sar_detection(
        sar_image(1)[:, :, 0],
        output_path=str(tmp_path),
        settings={"save_figure": True},
        **detection_args,
    )

    assert (tmp_path / f"{DATE}_S1.jpg").exists()


def test_displays_the_requested_channel(tmp_path, detection_args, monkeypatch):
    im_ms = sar_image(2)
    displayed = []
    monkeypatch.setattr(
        SDS_plotting,
        "_normalize_grayscale",
        lambda im: displayed.append(im) or np.zeros_like(im),
    )

    SDS_plotting.plot_sar_detection(
        im_ms,
        output_path=str(tmp_path),
        settings={"save_figure": False},
        band_index=1,
        **detection_args,
    )

    assert len(displayed) == 1
    assert np.array_equal(displayed[0], im_ms[:, :, 1])


def test_all_true_buffer_is_not_drawn(tmp_path, detection_args, monkeypatch):
    # a buffer covering the whole image means no reference shoreline was given, so
    # tinting the entire figure with it would only obscure the imagery
    detection_args["im_ref_buffer"] = np.ones((12, 10), dtype=bool)
    overlays = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "imshow", lambda self, im, **kw: overlays.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(2),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    assert not any(kw.get("cmap") == "PiYG" for kw in overlays)


def test_partial_buffer_is_drawn(tmp_path, detection_args, monkeypatch):
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    overlays = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "imshow", lambda self, im, **kw: overlays.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(2),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    assert any(kw.get("cmap") == "PiYG" for kw in overlays)


def axis_calls(monkeypatch):
    """Record which axis every imshow lands on, so the panels can be told apart."""
    calls = []

    def record(self, im, **kw):
        calls.append((self, im, kw))

    monkeypatch.setattr(SDS_plotting.plt.Axes, "imshow", record)
    return calls


def test_draws_three_panels(tmp_path, detection_args, monkeypatch):
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    calls = axis_calls(monkeypatch)

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    axes = list(dict.fromkeys(axis for axis, _, _ in calls))
    assert len(axes) == 3, "expected imagery, segmentation and buffer panels"


def test_only_the_third_panel_carries_the_buffer(tmp_path, detection_args, monkeypatch):
    # the middle panel has to stay readable, so the buffer goes on its own copy
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    calls = axis_calls(monkeypatch)

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    axes = list(dict.fromkeys(axis for axis, _, _ in calls))
    buffer_axes = {axis for axis, _, kw in calls if kw.get("cmap") == "PiYG"}

    assert buffer_axes == {axes[2]}, "the buffer must appear only on the third panel"


def test_the_middle_and_third_panels_both_show_the_segmentation(
    tmp_path, detection_args, monkeypatch
):
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    calls = axis_calls(monkeypatch)

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    axes = list(dict.fromkeys(axis for axis, _, _ in calls))
    # the class overlay is the one drawn with alpha=0.3
    class_axes = {axis for axis, _, kw in calls if kw.get("alpha") == 0.3}

    assert class_axes == {axes[1], axes[2]}
    assert axes[0] not in class_axes, "the first panel shows the imagery only"


def test_three_panels_without_a_reference_shoreline(tmp_path, detection_args, monkeypatch):
    # an all-True buffer constrains nothing; the panel still appears but is not tinted
    detection_args["im_ref_buffer"] = np.ones((12, 10), dtype=bool)
    calls = axis_calls(monkeypatch)

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    axes = list(dict.fromkeys(axis for axis, _, _ in calls))
    assert len(axes) == 3
    assert not any(kw.get("cmap") == "PiYG" for _, _, kw in calls)


def test_buffer_appears_in_the_legend(tmp_path, detection_args, monkeypatch):
    buffer = np.zeros((12, 10), dtype=bool)
    buffer[4:8, :] = True
    detection_args["im_ref_buffer"] = buffer
    legends = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "legend", lambda self, **kw: legends.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    labels = [handle.get_label() for handle in legends[0]["handles"]]
    assert "reference shoreline buffer" in labels


def test_the_buffer_swatch_matches_the_colour_actually_drawn():
    # the buffer is an all-True masked array under PiYG, which normalises to 0.0 and so
    # takes the PINK end of the colormap. A swatch showing the green end would send the
    # reader looking for something that is not on the image.
    buffer = np.zeros((6, 6), dtype=bool)
    buffer[2:4, :] = True
    masked = np.ma.masked_where(buffer == False, buffer)

    figure, axis = SDS_plotting.plt.subplots()
    image = axis.imshow(masked, cmap="PiYG", alpha=SDS_plotting.REF_BUFFER_ALPHA)
    drawn = matplotlib.colors.to_hex(image.cmap(image.norm(1.0)))
    SDS_plotting.plt.close(figure)

    assert drawn == SDS_plotting.REF_BUFFER_COLOR


def test_class_labels_appear_in_the_legend(tmp_path, detection_args, monkeypatch):
    # the two segmentation methods label opposite classes: Otsu marks the
    # high-backscatter class (land) as 1, the model marks water as 1
    figures = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "legend", lambda self, **kw: figures.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        class_labels=("land", "water"),
        **detection_args,
    )

    labels = [handle.get_label() for handle in figures[0]["handles"]]
    assert "land" in labels and "water" in labels
    assert "Otsu class 1" not in labels


def test_legend_defaults_to_the_otsu_labels(tmp_path, detection_args, monkeypatch):
    figures = []
    monkeypatch.setattr(
        SDS_plotting.plt.Axes, "legend", lambda self, **kw: figures.append(kw)
    )

    SDS_plotting.plot_sar_detection(
        sar_image(3),
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    labels = [handle.get_label() for handle in figures[0]["handles"]]
    assert "Otsu class 1" in labels and "Otsu class 2" in labels


def test_defaults_to_the_first_channel(tmp_path, detection_args, monkeypatch):
    im_ms = sar_image(2)
    displayed = []
    monkeypatch.setattr(
        SDS_plotting,
        "_normalize_grayscale",
        lambda im: displayed.append(im) or np.zeros_like(im),
    )

    SDS_plotting.plot_sar_detection(
        im_ms,
        output_path=str(tmp_path),
        settings={"save_figure": False},
        **detection_args,
    )

    assert np.array_equal(displayed[0], im_ms[:, :, 0])
