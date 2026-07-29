"""Tests for the reference shoreline buffer, including the no-reference-shoreline case."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat.SDS_shoreline import create_shoreline_buffer

IM_SHAPE = (40, 30)
PIXEL_SIZE = 10
EPSG = 32756
# a georeference whose world coordinates map 1:1 onto pixels at PIXEL_SIZE metres
GEOREF = np.array([500000.0, PIXEL_SIZE, 0.0, 6000000.0, 0.0, -PIXEL_SIZE])


def world_xy(col, row):
    """World coordinates of a pixel centre, for the georeference above."""
    return [GEOREF[0] + col * PIXEL_SIZE, GEOREF[3] - row * PIXEL_SIZE]


def vertical_reference_line():
    """A reference shoreline running down the middle of the image."""
    return np.array([world_xy(15, 5), world_xy(15, 35)])


def test_no_reference_shoreline_searches_the_whole_image():
    # regression: the dilation used to run outside the reference-shoreline branch,
    # raising UnboundLocalError whenever settings had no 'reference_shoreline'
    settings = {"output_epsg": EPSG, "max_dist_ref": 300}

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.shape == IM_SHAPE
    assert buffer.dtype == bool
    assert buffer.all()


def test_no_reference_shoreline_does_not_need_max_dist_ref():
    # max_dist_ref is only meaningful alongside a reference shoreline
    settings = {"output_epsg": EPSG}

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.all()


def test_reference_shoreline_restricts_the_buffer():
    settings = {
        "output_epsg": EPSG,
        "max_dist_ref": 30,  # 3 pixels
        "reference_shoreline": vertical_reference_line(),
    }

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.shape == IM_SHAPE
    assert buffer.any()
    assert not buffer.all()  # a buffer that covers everything would be pointless
    # the line itself is inside the buffer, the far corner is outside
    assert buffer[20, 15]
    assert not buffer[0, 0]


def test_buffer_widens_with_max_dist_ref():
    def buffer_for(max_dist_ref):
        settings = {
            "output_epsg": EPSG,
            "max_dist_ref": max_dist_ref,
            "reference_shoreline": vertical_reference_line(),
        }
        return create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer_for(20).sum() < buffer_for(60).sum()


def test_reference_shoreline_with_three_columns_is_accepted():
    # reference shorelines are often stored as x, y, z
    line = vertical_reference_line()
    settings = {
        "output_epsg": EPSG,
        "max_dist_ref": 30,
        "reference_shoreline": np.column_stack([line, np.zeros(len(line))]),
    }

    buffer = create_shoreline_buffer(IM_SHAPE, GEOREF, EPSG, PIXEL_SIZE, settings)

    assert buffer.any()
    assert not buffer.all()
