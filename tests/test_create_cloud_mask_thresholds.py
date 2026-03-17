import sys
import types
import warnings

import numpy as np


if "osgeo" not in sys.modules:
    osgeo_stub = types.ModuleType("osgeo")
    osgeo_stub.gdal = types.SimpleNamespace()
    osgeo_stub.osr = types.SimpleNamespace()
    sys.modules["osgeo"] = osgeo_stub

from coastsat import SDS_preprocess


def test_create_cloud_mask_keeps_size_40_removes_size_39_without_deprecation_warning():
    im_QA = np.zeros((30, 30), dtype=np.uint16)
    im_QA[1:4, 1:14] = 8  # 39 pixels (3x13)
    im_QA[10:15, 10:18] = 8  # 40 pixels (5x8)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cloud_mask = SDS_preprocess.create_cloud_mask(
            im_QA, satname="L8", cloud_mask_issue=False, collection="C02"
        )

    assert np.sum(cloud_mask) == 40
    assert cloud_mask[10:15, 10:18].all()
    assert not cloud_mask[1:4, 1:14].any()
    assert not any(
        issubclass(w.category, FutureWarning) for w in caught_warnings
    ), "create_cloud_mask should not emit FutureWarning from remove_small_objects."


def test_create_cloud_mask_issue_keeps_size_100_removes_size_99(monkeypatch):
    im_QA = np.zeros((50, 50), dtype=np.uint16)
    im_QA[1:10, 1:12] = 8  # 99 pixels (9x11)
    im_QA[20:30, 20:30] = 8  # 100 pixels (10x10)

    monkeypatch.setattr(
        SDS_preprocess.morphology, "binary_opening", lambda image, _: image
    )
    monkeypatch.setattr(
        SDS_preprocess.morphology, "square", lambda width: np.ones((width, width), dtype=bool)
    )

    cloud_mask = SDS_preprocess.create_cloud_mask(
        im_QA, satname="L8", cloud_mask_issue=True, collection="C02"
    )

    assert np.sum(cloud_mask) == 100
    assert cloud_mask[20:30, 20:30].all()
    assert not cloud_mask[1:10, 1:12].any()
