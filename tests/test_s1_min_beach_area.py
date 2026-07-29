"""Tests that the SAR Otsu path filters small objects in pixels, not square metres.

settings['min_beach_area'] is in m^2. The optical path divides it by pixel_size**2
before handing it to the classifier; the SAR Otsu path used to pass it straight through
to morphology.remove_small_objects, which counts pixels. At the 10 m S1 pixel that made
the filter 100x too aggressive and silently deleted real shorelines.
"""

import inspect
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_shoreline
from coastsat.SDS_shoreline import get_pixel_size_for_satellite, otsu_threshold


def image_with_a_small_and_a_large_bright_region():
    """A 20x20 image with a 9-pixel blob and a 200-pixel region above the threshold."""
    im = np.full((20, 20), -20.0)
    im[2:5, 2:5] = -5.0  # 9 pixels
    im[10:20, :] = -5.0  # 200 pixels
    return im


def test_the_argument_is_counted_in_pixels():
    im = image_with_a_small_and_a_large_bright_region()

    kept, _ = otsu_threshold(im, 4)

    assert kept[2:5, 2:5].all(), "a 9-pixel object must survive a 4-pixel minimum"
    assert kept[10:, :].all()


def test_objects_smaller_than_the_minimum_are_dropped():
    im = image_with_a_small_and_a_large_bright_region()

    kept, _ = otsu_threshold(im, 16)

    assert not kept[2:5, 2:5].any(), "a 9-pixel object must not survive a 16-pixel minimum"
    assert kept[10:, :].all(), "the large region must be untouched"


def test_a_float_minimum_is_accepted():
    # np.ceil(min_beach_area / pixel_size**2) is a float, and remove_small_objects
    # rejects a non-integer min_size
    im = image_with_a_small_and_a_large_bright_region()

    kept, _ = otsu_threshold(im, np.ceil(4500 / 10**2))  # 45 pixels

    assert not kept[2:5, 2:5].any()
    assert kept[10:, :].all()


def test_extract_shorelines_passes_pixels_to_the_sar_otsu_path():
    # guards against settings['min_beach_area'] (m^2) being passed straight through again
    source = inspect.getsource(SDS_shoreline.extract_shorelines)

    call = re.search(r"otsu_threshold\(([^)]*)\)", source)

    assert call is not None, "extract_shorelines no longer calls otsu_threshold"
    assert "min_beach_area_pixels" in call.group(1)
    assert 'settings["min_beach_area"]' not in call.group(1)


def test_the_conversion_matters_at_the_s1_pixel_size():
    # the bug was invisible in review because both values are "min beach area"; at 10 m
    # they differ by 100x, which is the whole point of converting
    pixel_size = get_pixel_size_for_satellite("S1")
    min_beach_area = 4500

    assert np.ceil(min_beach_area / pixel_size**2) == 45
    assert min_beach_area / np.ceil(min_beach_area / pixel_size**2) == 100
