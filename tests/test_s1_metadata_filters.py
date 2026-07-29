"""Tests that Sentinel-1 metadata properties become Earth Engine filters.

instrumentMode and orbitProperties_pass used to be read into sentinel_1_properties and
then silently ignored - setting them had no effect on the query. They now become
ee.Filter.eq filters on the S1 collection, and get_tier2_images no longer raises a
KeyError when S1 is in sat_list (S1 has no Tier 2 collection, like S2 and L9).
"""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat import SDS_download


class FakeFilter:
    """Stand-in for ee.Filter: the real one needs an initialised algorithm registry."""

    @staticmethod
    def eq(prop, value):
        return ("eq", prop, value)

    @staticmethod
    def listContains(prop, value):
        return ("listContains", prop, value)

    @staticmethod
    def And(*filters):
        return ("And", filters)


@pytest.fixture
def fake_ee_filter(monkeypatch):
    monkeypatch.setattr(SDS_download.ee, "Filter", FakeFilter)


# --------------------------------------------------- create_sentinel1_metadata_filters


def test_no_properties_means_no_filters(fake_ee_filter):
    assert SDS_download.create_sentinel1_metadata_filters({}) == []


def test_instrument_mode_becomes_an_eq_filter(fake_ee_filter):
    filters = SDS_download.create_sentinel1_metadata_filters({"instrumentMode": "IW"})

    assert filters == [("eq", "instrumentMode", "IW")]


def test_orbit_pass_becomes_an_eq_filter(fake_ee_filter):
    filters = SDS_download.create_sentinel1_metadata_filters(
        {"orbitProperties_pass": "DESCENDING"}
    )

    assert filters == [("eq", "orbitProperties_pass", "DESCENDING")]


def test_both_properties_yield_both_filters(fake_ee_filter):
    filters = SDS_download.create_sentinel1_metadata_filters(
        {"instrumentMode": "IW", "orbitProperties_pass": "ASCENDING"}
    )

    assert ("eq", "instrumentMode", "IW") in filters
    assert ("eq", "orbitProperties_pass", "ASCENDING") in filters


def test_values_are_normalised(fake_ee_filter):
    # GEE property values are upper case; a lower-case request must still match
    filters = SDS_download.create_sentinel1_metadata_filters(
        {"instrumentMode": " iw ", "orbitProperties_pass": "ascending"}
    )

    assert filters == [
        ("eq", "instrumentMode", "IW"),
        ("eq", "orbitProperties_pass", "ASCENDING"),
    ]


def test_polarizations_are_not_this_functions_business(fake_ee_filter):
    # they have their own AND filter, see create_polarization_filter
    filters = SDS_download.create_sentinel1_metadata_filters(
        {"transmitterReceiverPolarisation": ["VV", "VH"]}
    )

    assert filters == []


def test_rejects_an_invalid_instrument_mode(fake_ee_filter):
    with pytest.raises(ValueError, match="instrumentMode"):
        SDS_download.create_sentinel1_metadata_filters({"instrumentMode": "XX"})


def test_rejects_an_invalid_orbit_pass(fake_ee_filter):
    with pytest.raises(ValueError, match="orbitProperties_pass"):
        SDS_download.create_sentinel1_metadata_filters(
            {"orbitProperties_pass": "SIDEWAYS"}
        )


def test_the_default_properties_are_valid(fake_ee_filter):
    # the defaults must never be the thing that raises
    filters = SDS_download.create_sentinel1_metadata_filters(
        SDS_download.DEFAULT_SENTINEL_1_PROPERTIES
    )

    assert filters == [("eq", "instrumentMode", "IW")]


# ----------------------------------------------------------- wiring into the S1 query


def tier1_inputs(sentinel_1_properties=None):
    inputs = {
        "sat_list": ["S1"],
        "sitename": "site",
    }
    if sentinel_1_properties is not None:
        inputs["sentinel_1_properties"] = sentinel_1_properties
    return inputs


@pytest.fixture
def recorded_queries(monkeypatch):
    calls = []

    def fake_get_image_info(collection, satname, polygon, dates, **kwargs):
        calls.append({"collection": collection, "satname": satname, **kwargs})
        return []

    monkeypatch.setattr(SDS_download, "get_image_info", fake_get_image_info)
    return calls


def test_tier1_passes_the_merged_properties_to_the_query(recorded_queries):
    SDS_download.get_tier1_images(
        tier1_inputs({"orbitProperties_pass": "DESCENDING"}),
        polygon=[[[151.3, -33.7]]],
        dates=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    (call,) = recorded_queries
    # user values are merged over the defaults key by key
    assert call["sentinel_1_properties"]["orbitProperties_pass"] == "DESCENDING"
    assert call["sentinel_1_properties"]["instrumentMode"] == "IW"


def test_tier1_defaults_still_filter_the_instrument_mode(recorded_queries):
    SDS_download.get_tier1_images(
        tier1_inputs(),
        polygon=[[[151.3, -33.7]]],
        dates=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    (call,) = recorded_queries
    assert call["sentinel_1_properties"]["instrumentMode"] == "IW"
    assert "orbitProperties_pass" not in call["sentinel_1_properties"]


# -------------------------------------------------------------------- get_tier2_images


def test_tier2_skips_s1_instead_of_raising(recorded_queries):
    # col_names_T2 has no "S1" key; S1 + tier2 used to raise KeyError
    im_dict = SDS_download.get_tier2_images(
        {"sat_list": ["L5", "S1"], "sitename": "site"},
        polygon=[[[151.3, -33.7]]],
        dates_str=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    assert "S1" not in im_dict
    assert list(im_dict.keys()) == ["L5"]


def test_tier2_with_only_s1_returns_an_empty_dict(recorded_queries):
    im_dict = SDS_download.get_tier2_images(
        {"sat_list": ["S1"], "sitename": "site"},
        polygon=[[[151.3, -33.7]]],
        dates_str=["2023-01-01", "2023-02-01"],
        scene_cloud_cover=0.95,
        months_list=list(range(1, 13)),
    )

    assert im_dict == {}
    assert recorded_queries == []
