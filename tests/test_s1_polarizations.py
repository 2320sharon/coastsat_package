"""Tests for the Sentinel-1 polarization list and the Earth Engine polarization filter."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_download, SDS_tools


class FakeFilter:
    """Stand-in for ee.Filter: the real one needs an initialised algorithm registry."""

    @staticmethod
    def listContains(prop, value):
        return ("listContains", prop, value)

    @staticmethod
    def And(*filters):
        return ("And", filters)

    @staticmethod
    def Or(*filters):
        return ("Or", filters)


@pytest.fixture
def fake_ee_filter(monkeypatch):
    monkeypatch.setattr(SDS_download.ee, "Filter", FakeFilter)


def contains(polarization):
    return ("listContains", "transmitterReceiverPolarisation", polarization)


# ------------------------------------------------------------------ polarization list


def test_defaults_to_vv_and_vh():
    # both are downloaded by default, so the [VV, VH, VV-VH] composite is available
    assert SDS_download.get_sentinel1_polarizations({}) == ["VV", "VH"]


def test_default_comes_from_one_constant():
    assert list(SDS_tools.DEFAULT_SAR_POLARIZATIONS) == ["VV", "VH"]
    assert (
        SDS_download.DEFAULT_SENTINEL_1_PROPERTIES["transmitterReceiverPolarisation"]
        == list(SDS_tools.DEFAULT_SAR_POLARIZATIONS)
    )


def test_reads_requested_polarizations():
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": ["VV", "VH"]}}
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VV", "VH"]


def test_returns_canonical_order_regardless_of_request_order():
    # the channel order of im_ms must not depend on how the user listed them
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": ["VH", "VV"]}}
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VV", "VH"]


def test_normalizes_case_and_whitespace():
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": [" vh ", "vv"]}}
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VV", "VH"]


def test_dedupes():
    inputs = {
        "sentinel_1_properties": {"transmitterReceiverPolarisation": ["VH", "vh", "VH"]}
    }
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VH"]


def test_accepts_a_bare_string():
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": "VV"}}
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VV"]


def test_other_properties_do_not_drop_the_default_polarizations():
    # merging is key by key, so setting only instrumentMode keeps the default VV+VH
    inputs = {"sentinel_1_properties": {"instrumentMode": "EW"}}
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VV", "VH"]


def test_a_single_polarization_can_still_be_requested():
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": ["VH"]}}
    assert SDS_download.get_sentinel1_polarizations(inputs) == ["VH"]


def test_rejects_invalid_polarization():
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": ["VV", "XX"]}}
    with pytest.raises(ValueError, match="XX"):
        SDS_download.get_sentinel1_polarizations(inputs)


def test_rejects_empty_polarization_list():
    inputs = {"sentinel_1_properties": {"transmitterReceiverPolarisation": []}}
    with pytest.raises(ValueError, match="at least one polarization"):
        SDS_download.get_sentinel1_polarizations(inputs)


# ------------------------------------------------------------------ ee filter


def test_multiple_polarizations_are_anded(fake_ee_filter):
    # AND, not OR: every scene must carry all of them so the full set can be downloaded
    assert SDS_download.create_polarization_filter(["VV", "VH"]) == (
        "And",
        (contains("VV"), contains("VH")),
    )


def test_single_polarization_is_not_wrapped(fake_ee_filter):
    assert SDS_download.create_polarization_filter(["VH"]) == contains("VH")


def test_accepts_a_bare_string_polarization(fake_ee_filter):
    assert SDS_download.create_polarization_filter("VH") == contains("VH")


def test_or_mode_still_available(fake_ee_filter):
    assert SDS_download.create_polarization_filter(["VV", "VH"], mode="or") == (
        "Or",
        (contains("VV"), contains("VH")),
    )


def test_filter_rejects_invalid_polarization(fake_ee_filter):
    with pytest.raises(ValueError, match="XX"):
        SDS_download.create_polarization_filter(["XX"])


def test_filter_rejects_empty_list(fake_ee_filter):
    with pytest.raises(ValueError, match="(?i)at least one polarization"):
        SDS_download.create_polarization_filter([])


def test_filter_rejects_unknown_mode(fake_ee_filter):
    with pytest.raises(ValueError, match="mode"):
        SDS_download.create_polarization_filter(["VV", "VH"], mode="xor")


def test_valid_polarizations_come_from_one_place():
    assert set(SDS_tools.SAR_POLARIZATIONS) == {"VV", "VH", "HH", "HV"}
