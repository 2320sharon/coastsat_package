from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coastsat.SDS_shoreline import compute_cloud_metrics


def test_compute_cloud_metrics_returns_expected_values():
    cloud_mask = np.array([[True, False], [True, False]])
    im_nodata = np.array([[False, False], [True, False]])

    cloud_cover_combined, cloud_cover, cloud_mask_adv = compute_cloud_metrics(
        cloud_mask=cloud_mask,
        im_nodata=im_nodata,
    )

    expected_cloud_mask_adv = np.array([[True, False], [False, False]])

    assert cloud_cover_combined == pytest.approx(0.5)
    assert cloud_cover == pytest.approx(1 / 3)
    assert np.array_equal(cloud_mask_adv, expected_cloud_mask_adv)


def test_compute_cloud_metrics_raises_for_empty_nodata_list():
    cloud_mask = np.array([[True, False], [False, True]])

    with pytest.raises(ValueError, match="Cloud/no-data mask shape mismatch"):
        compute_cloud_metrics(cloud_mask=cloud_mask, im_nodata=[])