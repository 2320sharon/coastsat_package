import os

import numpy as np
import pytest
from tqdm.auto import tqdm

from coastsat import SDS_download, SDS_tools
from coastsat.SDS_preprocess import preprocess_single


def test_cloud_coverage_and_no_data_mask():
    polygon = [
        [
            [151.2957545, -33.7012561],
            [151.297557, -33.7388075],
            [151.312234, -33.7390216],
            [151.311204, -33.701399],
            [151.2957545, -33.7012561],
        ]
    ]
    polygon = SDS_tools.smallest_rectangle(polygon)

    dates = ["2023-12-01", "2024-01-01"]
    sat_list = ["L5", "L7", "L8", "L9", "S2"]
    collection = "C02"
    sitename = "test_cloud_filter_combined_90_cloud_80"
    filepath = os.path.join(os.getcwd(), "data")

    inputs = {
        "polygon": polygon,
        "dates": dates,
        "sat_list": sat_list,
        "sitename": sitename,
        "filepath": filepath,
        "landsat_collection": collection,
    }

    dataset_root = os.path.join(filepath, sitename)
    if not os.path.isdir(dataset_root):
        pytest.skip(f"Test data directory does not exist: {dataset_root}")

    metadata = SDS_download.get_metadata(inputs)

    max_cloud_no_data_cover = 0.9
    max_cloud_cover = 0.8
    do_cloud_mask = True
    cloud_mask_issue = False
    s2cloudless_prob = 60

    filtered_correct_flag = True

    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(inputs, satname)
        filenames = metadata[satname]["filenames"]

        for i in tqdm(
            range(len(filenames)), desc=f"{satname}: Analyzing imagery", leave=True, position=0
        ):
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)

            (
                im_ms,
                georef,
                cloud_mask,
                im_extra,
                im_QA,
                im_nodata,
            ) = preprocess_single(fn, satname, cloud_mask_issue, False, "C02", do_cloud_mask, s2cloudless_prob)

            cloud_cover_combined = np.sum(cloud_mask) / cloud_mask.size
            if cloud_cover_combined > max_cloud_no_data_cover:
                filtered_correct_flag = False

            cloud_mask_alone = np.logical_xor(cloud_mask, im_nodata)
            valid_pixels = np.sum(~im_nodata)
            cloud_cover = np.sum(cloud_mask_alone.astype(int)) / valid_pixels.astype(int)
            if cloud_cover > max_cloud_cover:
                filtered_correct_flag = False

    assert filtered_correct_flag, (
        "The cloud cover and no data mask is not working properly OR you used the incorrect "
        "thresholds that weren't the ones used to download the data"
    )