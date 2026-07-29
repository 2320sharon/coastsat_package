"""Tests for locating and swapping the polarization token in a SAR filename.

This is what pairs the VV and VH files of a scene, which live in separate folders
but otherwise share a name.
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat.SDS_tools import (
    get_polarization_from_filename,
    swap_polarization_in_filename,
)

DATE = "2023-12-03-19-15-49"


@pytest.mark.parametrize(
    "filename,expected",
    [
        (f"{DATE}_S1_site_VV.tif", "VV"),
        (f"{DATE}_S1_site_VH.tif", "VH"),
        (f"{DATE}_S1_site_HH.tif", "HH"),
        (f"{DATE}_S1_site_HV.tif", "HV"),
        # get_file_name() appends _dupN AFTER the polarization, so it is not last
        (f"{DATE}_S1_site_VV_dup0.tif", "VV"),
        (f"{DATE}_S1_site_VH_dup12.tif", "VH"),
        # optical products carry no polarization
        (f"{DATE}_S2_site_ms.tif", None),
        (f"{DATE}_L8_site_pan.tif", None),
    ],
)
def test_get_polarization_from_filename(filename, expected):
    assert get_polarization_from_filename(filename) == expected


def test_reads_the_token_from_a_full_path():
    path = str(Path("data") / "site" / "S1" / "VH" / f"{DATE}_S1_site_VH.tif")
    assert get_polarization_from_filename(path) == "VH"


def test_does_not_match_polarization_inside_sitename():
    # a site called 'myVHsite' must not be mistaken for a VH product
    filename = f"{DATE}_S1_myVHsite_VV.tif"
    assert get_polarization_from_filename(filename) == "VV"
    assert swap_polarization_in_filename(filename, "VH") == f"{DATE}_S1_myVHsite_VH.tif"


def test_swap_replaces_the_token():
    assert (
        swap_polarization_in_filename(f"{DATE}_S1_site_VV.tif", "VH")
        == f"{DATE}_S1_site_VH.tif"
    )


def test_swap_keeps_the_duplicate_suffix():
    assert (
        swap_polarization_in_filename(f"{DATE}_S1_site_VV_dup0.tif", "VH")
        == f"{DATE}_S1_site_VH_dup0.tif"
    )


def test_swap_to_the_same_polarization_is_a_no_op():
    filename = f"{DATE}_S1_site_VH.tif"
    assert swap_polarization_in_filename(filename, "VH") == filename


def test_swap_leaves_a_filename_without_a_token_alone():
    filename = f"{DATE}_S2_site_ms.tif"
    assert swap_polarization_in_filename(filename, "VH") == filename
