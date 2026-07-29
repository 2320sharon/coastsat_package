"""
This module contains all the functions needed to preprocess the satellite images
 before the shorelines can be extracted. This includes creating a cloud mask and
pansharpening/downsampling the multispectral bands.

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, datetime
from json import JSONEncoder

# image processing modules
import skimage.transform as transform
import skimage.morphology as morphology
import sklearn.decomposition as decomposition
import skimage.exposure as exposure
from skimage import img_as_ubyte
import imageio

# other modules
from osgeo import gdal
from pylab import ginput
import pickle
import geopandas as gpd
from shapely import geometry
import re
from shapely.geometry import Polygon
from typing import List, Dict, Any, Tuple, Union

# CoastSat modules
from coastsat import SDS_tools

np.seterr(all="ignore")  # raise/ignore divisions by 0 and nans


class SkipImageError(RuntimeError):
    """Raised when preprocessing determines that an image must be skipped."""


def preprocess_image(
    fn: Union[str, List[str]], satname: str, settings: Dict[str, Any], collection: str
) -> Tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    """
    Preprocesses satellite imagery depending on the satellite type and returns the processed outputs.

    Parameters:
    ----------
    fn : Union[str, List[str]]
        Filename or list of filenames of the satellite image(s). For Sentinel-1 (SAR), a list of one file is expected.
    satname : str
        Name of the satellite, e.g., "S1" for Sentinel-1 or "S2" for Sentinel-2.
    settings : Dict[str, Any]
        Dictionary containing preprocessing configuration. Expected keys include:
            - "apply_cloud_mask" (bool, optional): Whether to apply cloud masking (default is True).
            - "cloud_mask_issue" (Any): Issue-specific configuration for cloud masking.
            - "pan_off" (Any): Panchromatic settings.
            - "s2cloudless_prob" (int, optional): Threshold probability for s2cloudless cloud masking (default is 60).
    collection : str
        Satellite data collection name (used by optical processing).

    Returns:
    -------
    Tuple[np.ndarray, Any, np.ndarray, np.ndarray]
        - im_ms: Multispectral image array.
        - georef: Georeferencing metadata (varies by satellite).
        - cloud_mask: Boolean array indicating cloud presence (True = cloud).
        - im_nodata: Boolean array indicating no-data pixels (True = no data).
    """
    apply_cloud_mask = settings.get("apply_cloud_mask", True)

    if satname == "S1":
        # one file per polarization, stacked into one channel each
        im_ms, georef = SDS_tools.read_sar_image(fn)
        # when both polarizations are present, work on the [VV, VH, VV-VH] composite
        im_composite = SDS_tools.build_sar_composite(
            im_ms, [SDS_tools.get_polarization_from_filename(f) for f in fn]
        )
        if im_composite is not None:
            im_ms = im_composite
        im_nodata = np.zeros(im_ms.shape[:2], dtype=bool)
        cloud_mask = np.zeros(im_ms.shape[:2], dtype=bool)
        return im_ms, georef, cloud_mask, im_nodata
    else:
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(
            fn,
            satname,
            settings["cloud_mask_issue"],
            settings["pan_off"],
            collection,
            apply_cloud_mask,
            settings.get("s2cloudless_prob", 60),
        )
        return im_ms, georef, cloud_mask, im_nodata


def calculate_cloud_cover_combined(cloud_mask: np.ndarray) -> float:
    """
    Calculates the fraction of the image covered by clouds,
    including no-data pixels.
    """
    if cloud_mask is None or cloud_mask.size == 0:
        return 0.0

    # If both masks are all False, return 0.0
    if not cloud_mask.any():
        return 0.0

    total_pixels = cloud_mask.size
    return np.sum(cloud_mask) / total_pixels


def calculate_cloud_cover_excluding_nodata(
    cloud_mask: np.ndarray, nodata_mask: np.ndarray
) -> float:
    """
    Calculates cloud cover excluding no-data pixels.

    Args:
        cloud_mask: Boolean array where True indicates cloud pixels.
        nodata_mask: Boolean array where True indicates no-data pixels.

    Returns:
        Cloud cover as a float between 0 and 1.
    """
    # Handle empty or None inputs
    if (
        cloud_mask is None
        or nodata_mask is None
        or cloud_mask.size == 0
        or nodata_mask.size == 0
    ):
        return 0.0

    # If both masks are all False, return 0.0
    if not cloud_mask.any() and not nodata_mask.any():
        return 0.0

    valid_pixels = ~nodata_mask
    valid_pixel_count = np.sum(valid_pixels)

    if valid_pixel_count == 0:
        return 1.0  # fallback: treat as fully cloudy if no valid pixels

    clean_cloud_mask = np.logical_and(cloud_mask, valid_pixels)
    return np.sum(clean_cloud_mask.astype(np.float32)) / valid_pixel_count


def filter_images_by_cloud_cover_nodata(
    fn,
    satname,
    cloud_mask_issue,
    max_cloud_no_data_cover=None,
    max_cloud_cover=None,
    do_cloud_mask=True,
    s2cloudless_prob=60,
    return_metrics=False,
):
    """
    Filters images based on cloud cover and no data pixels.

    Args:
        fn (list[str]): A list of file paths.
            The first file path is the full path to mulitspectral image (ms)
            The second path is the swir/panchromatic image (swir/pan) depending on the satellite.
                - For S2 the second path is the swir image (swir)
                - For Landsat the second path is the panchromatic image (pan) except for Landsat 5 where it is the cloud mask image (mask)
            The third path is the cloud mask image (mask)
                - except on Landsat 5 where it is the second path
        satname (str): Name of the satellite.
        cloud_mask_issue (bool): True if there is an issue with the cloud mask and sand pixels are being masked on the images
        max_cloud_no_data_cover (float, optional): Maximum cloud and no data cover threshold. Defaults to None.
        max_cloud_cover (float, optional): Maximum cloud cover threshold. Defaults to None.
        do_cloud_mask (bool, optional): Flag indicating if cloud masking should be performed. Defaults to True.
        s2cloudless_prob (int, optional): Probability threshold for cloud masking. Defaults to 60.

    Returns:
        bool: True if the image is filtered, False otherwise.
        If return_metrics is True, returns a dict with keys:
            - filtered (bool)
            - reason (str)
            - cloud_cover_combined (float)
            - cloud_cover (float)
    """
    result = {
        "filtered": False,
        "reason": "none",
        "cloud_cover_combined": 0.0,
        "cloud_cover": 0.0,
    }

    if max_cloud_no_data_cover is None and max_cloud_cover is None:
        # nothing was filtered return False
        return result if return_metrics else False
    (
        im_ms,
        georef,
        cloud_mask,
        im_extra,
        im_QA,
        im_nodata,
    ) = preprocess_single(
        fn, satname, cloud_mask_issue, False, "C02", do_cloud_mask, s2cloudless_prob
    )

    # check if the cloud and no data mask exceed the threshold
    cloud_cover_combined = calculate_cloud_cover_combined(cloud_mask)
    cloud_cover = calculate_cloud_cover_excluding_nodata(cloud_mask, im_nodata)
    result["cloud_cover_combined"] = cloud_cover_combined
    result["cloud_cover"] = cloud_cover

    exceeded_reasons = []
    exceeds_cloud_no_data = (
        max_cloud_no_data_cover is not None
        and cloud_cover_combined > max_cloud_no_data_cover
    )
    exceeds_cloud = max_cloud_cover is not None and cloud_cover > max_cloud_cover

    if exceeds_cloud_no_data:
        exceeded_reasons.append(
            f"percent_no_data (cloud_cover_combined) > {max_cloud_no_data_cover}"
        )
    if exceeds_cloud:
        exceeded_reasons.append(f"cloud_cover > {max_cloud_cover}")

    if exceeded_reasons:
        result["reason"] = "; ".join(exceeded_reasons)

    # if cloud+no-data threshold is exceeded, remove image
    if exceeds_cloud_no_data:
        were_images_filtered = remove_files_above_threshold(
            fn,
            cloud_cover_combined,
            max_cloud_no_data_cover,
            msg="due to cloud and no data cover",
        )
        # return True if the image was filtered otherwise continue to check the cloud cover
        if were_images_filtered:
            result["filtered"] = True
            return result if return_metrics else True

    # otherwise, if cloud-only threshold is exceeded, remove image
    if exceeds_cloud:
        result["filtered"] = remove_files_above_threshold(
            fn, cloud_cover, max_cloud_cover, msg="due to cloud cover"
        )
        return result if return_metrics else result["filtered"]

    # if nothing was filtered then return False
    return result if return_metrics else False


def remove_files_above_threshold(
    filepaths: list[str], prc: float, threshold: int, msg: str = "due to cloud cover"
):
    """
    Removes files that exceed the cloud cover threshold.

    Args:
        filepaths (list[str]): A list of file paths.
            The first file path is the full path to mulitspectral image (ms)
            The second path is the swir/panchromatic image (swir/pan) depending on the satellite.
                - For S2 the second path is the swir image (swir)
                - For Landsat the second path is the panchromatic image (pan) except for Landsat 5 where it is the cloud mask image (mask)
            The third path is the cloud mask image (mask)
                - except on Landsat 5 where it is the second path
        prc (float): The percentage of cloud cover.
        threshold (float): The threshold value for cloud cover.
        msg (str): The message to be printed when the threshold is exceeded.
        ex. "due to cloud cover" will print "skipping image due to cloud cover 50.0%"

    Returns:
        bool: True if files were removed, False otherwise.
    """
    if prc > threshold:
        print(
            f"skipping image '{os.path.basename(filepaths[0])}' {msg} {prc*100:.2f}% exceeds threshold of {threshold*100:.2f}%"
        )
        # delete files that exceed the cloud cover threshold
        for file in filepaths:
            os.remove(file)
        return True
    return False


def create_gdf_from_image_extent(
    height: int, width: int, georef: np.array, image_epsg: int, output_epsg: int
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame containing a polygon from the extent of an image.

    Args:
        height (int): The height of the image in pixels.
        width (int): The width of the image in pixels.
        georef (np.array): vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg (int): The EPSG code of the image's coordinate reference system.
        output_epsg (int): The EPSG code of the desired output coordinate reference system.

    Returns:
        gdf (GeoDataFrame): A GeoDataFrame representing the extent of the image.

    """
    rectangle_coords = [(0, 0), (0, width), (height, width), (height, 0), (0, 0)]
    points = np.array(rectangle_coords)
    # convert pixel coordinates to world coordinates
    world_img = SDS_tools.convert_pix2world(points, georef)
    image_bounds_in_epsg = SDS_tools.convert_epsg(world_img, image_epsg, output_epsg)
    # drop the z coordinate
    image_bounds_in_epsg = image_bounds_in_epsg[:, :2]
    # Reverse the order of each point
    if output_epsg == 4326 or output_epsg == 4327:
        image_bounds_in_epsg = image_bounds_in_epsg[:, ::-1]
    # create a geodataframe
    gdf = gpd.GeoDataFrame(
        {"geometry": [Polygon(image_bounds_in_epsg)]},
        crs=output_epsg,
    )
    return gdf


def convert_coords_to_gdf(coords: np.ndarray, epsg: int) -> gpd.GeoDataFrame:
    """
    Converts pixel coordinates to a GeoDataFrame with the same coordinates in the reference system.

    Parameters:
    - coords (np.ndarray): A 2D array containing the pixel coordinates.
    - georef (list): A list containing the georeferencing information.
    - epsg (int): The EPSG code of the reference system.

    Returns:
    - gpd.GeoDataFrame: A GeoDataFrame containing the pixel coordinates in the reference system.
    """

    # Convert the polygon coordinates to a valid geometry object
    shapely_polygon = Polygon(coords[0])

    # Create a GeoDataFrame with the converted polygon
    gdf = gpd.GeoDataFrame(geometry=[shapely_polygon], crs=epsg)

    # Convert the coordinates to the reference system
    gdf = gdf.to_crs(epsg=epsg)

    return gdf


def bind_image_size(
    array: np.ndarray, im_shape: tuple, input_epsg: int, georef: list, output_epsg: int
):
    """
    Binds the image size to the reference shoreline coordinates, adjusting coordinates outside the image
    to the nearest edge within the image boundaries.

    Args:
        array (numpy.ndarray): The reference shoreline coordinates.
        im_shape (tuple): The shape of the image.
        input_epsg (int): The EPSG code of the input coordinates.
        georef (np.array): vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        output_epsg (int): The EPSG code of the image.

    Returns:
        numpy.ndarray: The adjusted reference shoreline coordinates inside or on the edge of the image.
    """
    # Convert array to pixel coordinates
    ref_sl_conv = SDS_tools.convert_epsg(array, input_epsg, output_epsg)[:, :-1]
    pixel_coords = SDS_tools.convert_world2pix(ref_sl_conv, georef)

    pixel_coords[:, 0] = np.clip(pixel_coords[:, 0], 0, im_shape[1] - 1)
    pixel_coords[:, 1] = np.clip(pixel_coords[:, 1], 0, im_shape[0] - 1)

    return pixel_coords


def transform_world_coords_to_pixel_coords(
    array: np.ndarray, input_epsg: int, georef: np.array, output_epsg: int
) -> np.array:
    """
    Transforms world coordinates to pixel coordinates.

    Args:
        array (np.ndarray): The input array containing world coordinates.
        input_epsg (int): The EPSG code of the input coordinate system.
        georef (np.array): vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        output_epsg (int): The EPSG code of the output coordinate system.

    Returns:
        np.ndarray: The pixel coordinates corresponding to the input world coordinates.
    """
    # Convert array to pixel coordinates
    ref_sl_conv = SDS_tools.convert_epsg(array, input_epsg, output_epsg)[:, :-1]
    pixel_coords = SDS_tools.convert_world2pix(ref_sl_conv, georef)

    return pixel_coords


def find_edge_padding(im_band: np.ndarray) -> np.ndarray:
    """
    Finds the padding required for each edge of an image band based on the presence of data.

    Parameters:
    im_band (numpy.ndarray): The image band.

    Returns:
    tuple: A tuple containing the top, bottom, left, and right padding values.
    """

    # Assuming non-data values are zeros. Adjust the condition if needed.
    is_data = im_band != 0

    # Function to find padding for one edge
    def find_edge_data(is_data_along_edge):
        for idx, has_data in enumerate(is_data_along_edge):
            if has_data:
                return idx
        return len(is_data_along_edge)  # Return full length if no data found

    # Calculate padding for each side
    top_padding = find_edge_data(np.any(is_data, axis=1))
    bottom_padding = find_edge_data(np.any(is_data, axis=1)[::-1])
    left_padding = find_edge_data(np.any(is_data, axis=0))
    right_padding = find_edge_data(np.any(is_data, axis=0)[::-1])

    return top_padding, bottom_padding, left_padding, right_padding


def pad_edges(im_swir: np.ndarray, im_nodata: np.ndarray) -> np.ndarray:
    """
    Adds 0's located along the edges of im_swir to the nodata array.

    Fixes the issue where 0s are created along the edges of the SWIR1 band caused by reprojecting the 20 m band onto the 10m pixel grid (with bilinear interpolation in GDAL)

    Args:
        im_swir (np.ndarray): The SWIR image.
        im_nodata (np.ndarray): The nodata array.

    Returns:
        np.ndarray: The nodata array with padded edges.
    """
    # check if the two images are the same shape and if not then return the original nodata array
    # Typically im_swir has shape (x,y,1) and im_nodata has shape (x,y)

    # if im_swir.shape != im_nodata.shape:
    #     return im_nodata

    top_pad, bottom_pad, left_pad, right_pad = find_edge_padding(im_swir)

    # if bottom pad is 0 the entire image gets set to True
    if bottom_pad > 0:
        im_nodata[-bottom_pad:, :] = True
    # if right pad is 0 the entire image gets set to True
    if right_pad > 0:
        im_nodata[:, -right_pad:] = True

    # Apply this padding to your masks
    im_nodata[:top_pad, :] = True
    im_nodata[:, :left_pad] = True
    return im_nodata


def write_to_json(filepath: str, settings: dict):
    """ "Write the  settings dictionary to json file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    to_file(settings, filepath)


def to_file(data: dict, filepath: str) -> None:
    """
    Serializes a dictionary to a JSON file, handling special serialization for datetime and numpy ndarray objects.

    The function handles two special cases:
    1. If the data contains datetime.date or datetime.datetime objects, they are serialized to their ISO format.
    2. If the data contains numpy ndarray objects, they are converted to lists before serialization.

    Parameters:
    - data (dict): Dictionary containing the data to be serialized to a JSON file.
    - filepath (str): Path (including filename) where the JSON file should be saved.

    Returns:
    - None

    Note:
    - This function requires the json, datetime, and numpy modules to be imported.
    """

    class DateTimeEncoder(JSONEncoder):
        # Override the default method
        def default(self, obj):
            if isinstance(obj, (date, datetime)):
                return obj.isoformat()
            if isinstance(obj, (np.ndarray)):
                new_obj = [array.tolist() for array in obj]
                return new_obj

    with open(filepath, "w") as fp:
        json.dump(data, fp, cls=DateTimeEncoder)


def read_bands(filename: str, satname: str = "") -> list[np.ndarray]:
    """
    Read all the raster bands of a geospatial image file using GDAL.

    This function opens the provided geospatial file using GDAL in read-only mode
    and extracts each of the raster bands as a separate array. Each array represents
    the values of the raster band across the entire spatial extent of the image.

    Parameters:
    -----------
    filename : str
        The path to the geospatial image file (e.g., a GeoTIFF) to be read.

    Returns:
    --------
    list of np.ndarray
        A list containing 2D numpy arrays. Each array in the list represents
        one band from the geospatial image. The order of the arrays in the list
        corresponds to the order of the bands in the original image.

    Notes:
    ------
    This function relies on GDAL for reading geospatial data. Ensure that GDAL
    is properly installed and available in the Python environment.

    Example:
    --------
    bands = read_bands('path/to/image.tif')
    """
    data = gdal.Open(filename, gdal.GA_ReadOnly)
    # save the contents of each raster band as an array and save each array to the bands list
    # save separate NumPy array for each raster band in the dataset, with each array representing the pixel values of the corresponding band
    if satname == "S2" and data.RasterCount == 5:
        bands = [
            data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount - 1)
        ]
    else:
        bands = [
            data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)
        ]
    return bands


def get_nodata_mask(im_ms: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Generate a mask indicating no-data values in a multi-band image.

    The function identifies pixels in each band of the input image that contain
    values of -inf or NaN and returns a binary mask marking those pixels.

    Parameters:
    -----------
    im_ms : np.ndarray
        A 3D numpy array representing the multi-band image where the last
        dimension represents different bands.

    shape : tuple
        The shape of the output binary mask, typically the shape of
        the spatial dimensions (height, width) of the input image.

    Returns:
    --------
    np.ndarray
        A binary mask (of the provided shape) with the same spatial dimensions
        as the input image. Pixels with no-data values in any band in the input
        image are marked as True in the mask, and others are marked as False.

    Notes:
    ------
    The function checks for -inf and NaN values across all bands for each pixel
    and considers a pixel as no-data even if only one of its bands has a no-data value.
    """
    # check if -inf or nan values on any band and eventually add those pixels to cloud mask
    im_nodata = np.zeros(shape).astype(bool)
    for k in range(im_ms.shape[2]):
        im_inf = np.isin(im_ms[:, :, k], -np.inf)
        im_nan = np.isnan(im_ms[:, :, k])
        im_nodata = np.logical_or(np.logical_or(im_nodata, im_inf), im_nan)
    return im_nodata


def get_zero_pixels(im_ms: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Generate a mask indicating pixels with zero values in specific bands of a multi-band image.

    The function identifies pixels that have a value of 0 in the Green, NIR, and SWIR bands
    of the input image and returns a binary mask marking those pixels.

    Parameters:
    -----------
    im_ms : np.ndarray
        A 3D numpy array representing the multi-band image where the last
        dimension represents different bands.

    shape : tuple
        The shape of the output binary mask, typically the shape of
        the spatial dimensions (height, width) of the input image.

    Returns:
    --------
    np.ndarray
        A binary mask (of the provided shape) with the same spatial dimensions
        as the input image. Pixels with zero values in the Green, NIR, and SWIR bands
        of the input image are marked as True in the mask, and others are marked as False.

    Notes:
    ------
    The function specifically checks for zero values in the Green (band index 1),
    NIR (band index 3), and SWIR (band index 4) bands of the input image. A pixel is
    considered to be a zero pixel only if all of these three bands have a zero value for that pixel.
    """
    # Identify identify pixels with 0 intensity in the Green, NIR and SWIR bands
    im_zeros = np.ones(shape).astype(bool)
    for k in [1, 3, 4]:  # loop through the Green, NIR and SWIR bands
        im_zeros = np.logical_and(np.isin(im_ms[:, :, k], 0), im_zeros)
    return im_zeros


def raise_exception_if_file_is_empty(filepath: str, msg=""):
    """
    Raises an exception if the file is empty.

    Args:
        filepath (str): The path to the file to check.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"\nFile '{filepath}' does not exist.{msg}")


# Main function to preprocess a satellite image (L5, L7, L8, L9 or S2)
def preprocess_single(
    fn,
    satname,
    cloud_mask_issue,
    pan_off,
    collection="C02",
    do_cloud_mask=True,
    s2cloudless_prob=60,
):
    """
    Reads the image and outputs the pansharpened/down-sampled multispectral bands,
    the georeferencing vector of the image (coordinates of the upper left pixel),
    the cloud mask, the QA band and a no_data image.
    For Landsat 7-8 it also outputs the panchromatic band and for Sentinel-2 it
    also outputs the 20m SWIR band.

    KV WRL 2018

    Arguments:
    -----------
    fn: str or list of str
        filename of the .TIF file containing the image. For L7, L8 and S2 this
        is a list of filenames, one filename for each band at different
        resolution (30m and 15m for Landsat 7-8, 10m, 20m, 60m for Sentinel-2)
    satname: str
        name of the satellite mission (e.g., 'L5')
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being masked on the images
    pan_off : boolean
        if True, disable panchromatic sharpening and ignore pan band
    collection: str
        Landsat collection 'C02'
    do_cloud_mask: boolean
        if True, apply the cloud mask to the image. If False, the cloud mask is not applied.
    s2cloudless_prob: float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask

    Returns:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale] defining the
        coordinates of the top-left pixel of the image
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_extra : np.array
        2D array containing the 20m resolution SWIR band for Sentinel-2 and the 15m resolution
        panchromatic band for Landsat 7 and Landsat 8. This field is empty for Landsat 5.
    im_QA: np.array
        2D array containing the QA band, from which the cloud_mask can be computed.
    im_nodata: np.array
        2D array with True where no data values (-inf) are located

    """

    if isinstance(fn, list):
        fn_to_split = fn[0]
    elif isinstance(fn, str):
        fn_to_split = fn
    # split by os.sep and only get the filename at the end then split again to remove file extension
    fn_to_split = fn_to_split.split(os.sep)[-1].split(".")[0]
    # search for the year the tif was taken with regex and convert to int
    year = int(re.search("[0-9]+", fn_to_split).group(0))
    # after 2022 everything is automatically from Collection 2
    if collection == "C01" and year >= 2022:
        collection = "C02"

    # =============================================================================================#
    # L5 images
    # =============================================================================================#
    if satname == "L5":
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_mask = fn[1]
        raise_exception_if_file_is_empty(fn_ms)
        raise_exception_if_file_is_empty(fn_mask)
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = read_bands(fn_ms)
        im_ms = np.stack(bands, 2)
        # read cloud mask
        im_QA = read_bands(fn_mask)[0]
        if not do_cloud_mask:
            cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = get_nodata_mask(im_ms, cloud_mask.shape)
            # cloud mask is the same as the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
        else:
            cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = get_nodata_mask(im_ms, cloud_mask.shape)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # no extra image for Landsat 5 (they are all 30 m bands)
        im_extra = []
    # =============================================================================================#
    # L7, L8 and L9 images
    # =============================================================================================#
    elif satname in ["L7", "L8", "L9"]:
        # filepaths to .tif files
        fn_ms = fn[0]
        fn_pan = fn[1]
        fn_mask = fn[2]
        raise_exception_if_file_is_empty(fn_ms)
        raise_exception_if_file_is_empty(fn_mask)
        raise_exception_if_file_is_empty(fn_pan)
        # read ms bands
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = read_bands(fn_ms)
        im_ms = np.stack(bands, 2)
        # read cloud mask and get the QA from the first band
        im_QA = read_bands(fn_mask)[0]

        if not do_cloud_mask:
            cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = get_nodata_mask(im_ms, cloud_mask.shape)
            # cloud mask is the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
        else:
            cloud_mask = create_cloud_mask(im_QA, satname, cloud_mask_issue, collection)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = get_nodata_mask(im_ms, cloud_mask.shape)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)

        # if panchromatic sharpening is turned off
        if pan_off:
            # ms bands are untouched and the extra image is empty
            im_extra = []

        # otherwise perform panchromatic sharpening
        else:
            # read panchromatic band
            data = gdal.Open(fn_pan, gdal.GA_ReadOnly)
            georef = np.array(data.GetGeoTransform())
            bands = [
                data.GetRasterBand(k + 1).ReadAsArray() for k in range(data.RasterCount)
            ]
            im_pan = bands[0]

            # pansharpen Green, Blue, NIR for Landsat 7
            if satname == "L7":
                try:
                    im_ms_ps = pansharpen(im_ms[:, :, [1, 2, 3]], im_pan, cloud_mask)
                except:  # if pansharpening fails, keep downsampled bands (for long runs)
                    print("\npansharpening of image %s failed." % fn[0])
                    im_ms_ps = im_ms[:, :, [1, 2, 3]]
                # add downsampled Blue and SWIR1 bands
                im_ms_ps = np.append(im_ms[:, :, [0]], im_ms_ps, axis=2)
                im_ms_ps = np.append(im_ms_ps, im_ms[:, :, [4]], axis=2)
                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan

            # pansharpen Blue, Green, Red for Landsat 8 and 9
            elif satname in ["L8", "L9"]:
                try:
                    im_ms_ps = pansharpen(im_ms[:, :, [0, 1, 2]], im_pan, cloud_mask)
                except:  # if pansharpening fails, keep downsampled bands (for long runs)
                    print("\npansharpening of image %s failed." % fn[0])
                    im_ms_ps = im_ms[:, :, [0, 1, 2]]
                # add downsampled NIR and SWIR1 bands
                im_ms_ps = np.append(im_ms_ps, im_ms[:, :, [3, 4]], axis=2)
                im_ms = im_ms_ps.copy()
                # the extra image is the 15m panchromatic band
                im_extra = im_pan

    # =============================================================================================#
    # S2 images
    # =============================================================================================#
    if satname == "S2":
        # read 10m bands (R,G,B,NIR)
        fn_ms = fn[0]
        # read 20m band (SWIR1) from the first band
        fn_swir = fn[1]
        fn_mask = fn[2]
        raise_exception_if_file_is_empty(fn_ms)
        raise_exception_if_file_is_empty(fn_mask)
        raise_exception_if_file_is_empty(fn_swir)
        data = gdal.Open(fn_ms, gdal.GA_ReadOnly)
        georef = np.array(data.GetGeoTransform())
        bands = read_bands(fn_ms, satname)
        im_ms = np.stack(bands, 2)
        im_ms = im_ms / 10000  # TOA scaled to 10000
        # read s2cloudless cloud probability (last band in ms image)
        cloud_prob = data.GetRasterBand(data.RasterCount).ReadAsArray()

        # if image contains only zeros (can happen with S2), skip the image
        if sum(sum(sum(im_ms))) < 1:
            # Raise a skip-specific exception so callers can continue the batch
            # without receiving malformed placeholder arrays.
            raise SkipImageError(
                f"Skipped image because Sentinel-2 multispectral file contains only zeros: '{fn_ms}'"
            )

        im_swir = read_bands(fn_swir)[0] / 10000  # TOA scaled to 10000
        im_swir = np.expand_dims(im_swir, axis=2)

        # append down-sampled SWIR1 band to the other 10m bands
        im_ms = np.append(im_ms, im_swir, axis=2)

        # create cloud mask using 60m QA band (not as good as Landsat cloud cover)
        fn_mask = fn[2]
        im_QA = read_bands(fn_mask)[0]
        if not do_cloud_mask:
            cloud_mask_QA60 = create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # compute cloud mask using s2cloudless probability band
            cloud_mask_s2cloudless = create_s2cloudless_mask(
                cloud_prob, s2cloudless_prob
            )
            # combine both cloud masks
            cloud_mask = np.logical_or(cloud_mask_QA60, cloud_mask_s2cloudless)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = get_nodata_mask(im_ms, cloud_mask.shape)
            im_nodata = pad_edges(im_swir, im_nodata)
            # cloud mask is the no data mask
            cloud_mask = im_nodata.copy()
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            if "merged" in fn_ms:
                im_nodata = morphology.dilation(im_nodata, SDS_tools.footprint_square(5))
            # update cloud mask with all the nodata pixels
            # v0.1.40 change : might be bug
            cloud_mask = np.logical_or(cloud_mask, im_nodata)
        else:  # apply the cloud mask
            # compute cloud mask using QA60 band
            cloud_mask_QA60 = create_cloud_mask(
                im_QA, satname, cloud_mask_issue, collection
            )
            # compute cloud mask using s2cloudless probability band
            cloud_mask_s2cloudless = create_s2cloudless_mask(
                cloud_prob, s2cloudless_prob
            )
            # combine both cloud masks
            cloud_mask = np.logical_or(cloud_mask_QA60, cloud_mask_s2cloudless)
            # add pixels with -inf or nan values on any band to the nodata mask
            im_nodata = get_nodata_mask(im_ms, cloud_mask.shape)
            im_nodata = pad_edges(im_swir, im_nodata)
            # check if there are pixels with 0 intensity in the Green, NIR and SWIR bands and add those
            # to the cloud mask as otherwise they will cause errors when calculating the NDWI and MNDWI
            im_zeros = get_zero_pixels(im_ms, cloud_mask.shape)
            # add zeros to im nodata
            im_nodata = np.logical_or(im_zeros, im_nodata)
            # update cloud mask with all the nodata pixels
            cloud_mask = np.logical_or(cloud_mask, im_nodata)
            if "merged" in fn_ms:
                im_nodata = morphology.dilation(im_nodata, SDS_tools.footprint_square(5))
            # move cloud mask to above if statement to avoid bug in v0.1.40

        # no extra image
        im_extra = []

    return im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def create_cloud_mask(im_QA, satname, cloud_mask_issue, collection):
    """
    Creates a cloud mask using the information contained in the QA band.

    KV WRL 2018

    Arguments:
    -----------
    im_QA: np.array
        Image containing the QA band
    satname: string
        short name for the satellite: ```'L5', 'L7', 'L8' or 'S2'```
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels are being
        erroneously masked on the images
    collection: str
        Landsat collection 'C02' or deprecated 'C01'

    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise

    """
    if satname == "S2":
        # 1024 = dense cloud, 2048 = cirrus clouds
        cloud_values = [1024, 2048]
    else:
        if collection == "C01":
            if satname in ["L8", "L9"]:
                # 2800, 2804, 2808, 2812 = High confidence cloud
                # 6896, 6900, 6904, 6908 = High confidence cirrus cloud
                cloud_values = [2800, 2804, 2808, 2812, 6896, 6900, 6904, 6908]
            elif satname in ["L4", "L5", "L7", "L8"]:
                # 752, 756, 760, 764 = High confidence cloud
                cloud_values = [752, 756, 760, 764]
        elif collection == "C02":
            # function to return flag for n-th bit
            def is_set(x, n):
                return x & 1 << n != 0

            # dilated cloud = bit 1
            # cirrus = bit 2
            # cloud = bit 3
            qa_values = np.unique(im_QA.flatten())
            cloud_values = []
            for qaval in qa_values:
                for k in [1, 2, 3]:  # check the first 3 flags
                    if is_set(qaval, k):
                        cloud_values.append(qaval)

    # find which pixels have bits corresponding to cloud values
    cloud_mask = np.isin(im_QA, cloud_values)

    # remove cloud pixels that form very thin features. These are beach or swash pixels that are
    # erroneously identified as clouds by the CFMASK algorithm applied to the images by the USGS.
    if sum(sum(cloud_mask)) > 0 and sum(sum(~cloud_mask)) > 0:
        cloud_mask = morphology.remove_small_objects(
            cloud_mask, min_size=40, connectivity=1
        )
    # this is only for if the cloud mask was applied to the shoreline accidentally
    if cloud_mask_issue:
        cloud_mask = np.zeros_like(im_QA, dtype=bool)
        for value in cloud_values:
            cloud_mask_temp = np.isin(im_QA, value)
            elem = SDS_tools.footprint_square(6)  # use a square of width 6 pixels
            cloud_mask_temp = SDS_tools.binary_opening(
                cloud_mask_temp, elem
            )  # perform image opening
            cloud_mask_temp = morphology.remove_small_objects(
                cloud_mask_temp, min_size=100, connectivity=1
            )
            cloud_mask = np.logical_or(cloud_mask, cloud_mask_temp)

    return cloud_mask


def create_s2cloudless_mask(cloud_prob, s2cloudless_prob=60):
    """
    Creates a cloud mask using the s2cloudless band.

    KV WRL 2023

    Arguments:
    -----------
    cloud_prob: np.array
        Image containing the s2cloudless cloud probability
    s2cloudless_prob: float [0,100)
        threshold to identify cloud pixels in the s2cloudless probability mask. Default is 60.
        If the cloud probability is above this threshold, the pixel is classified as cloudy.

    Returns:
    -----------
    cloud_mask : np.array
        boolean array with True if a pixel is cloudy and False otherwise

    """
    # find which pixels have bits corresponding to cloud values
    cloud_mask = cloud_prob > s2cloudless_prob
    # dilate cloud mask
    elem = SDS_tools.footprint_square(6)  # use a square of width 6 pixels
    cloud_mask = SDS_tools.binary_opening(cloud_mask, elem)  # perform image opening

    return cloud_mask


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram matches
    that of a target image.

    Arguments:
    -----------
    source: np.array
        Image to transform; the histogram is computed over the flattened
        array
    template: np.array
        Template image; can have different dimensions to source

    Returns:
    -----------
    matched: np.array
        The transformed output image

    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True
    )
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def pansharpen(im_ms, im_pan, cloud_mask):
    """
    Pansharpens a multispectral image, using the panchromatic band and a cloud mask.
    A PCA is applied to the image, then the 1st PC is replaced, after histogram
    matching with the panchromatic band. Note that it is essential to match the
    histrograms of the 1st PC and the panchromatic band before replacing and
    inverting the PCA.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Multispectral image to pansharpen (3D)
    im_pan: np.array
        Panchromatic band (2D)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:
    -----------
    im_ms_ps: np.ndarray
        Pansharpened multispectral image (3D)

    """
    # check that cloud cover is not too high otherwise pansharpening fails
    if sum(sum(cloud_mask)) > 0.95 * cloud_mask.shape[0] * cloud_mask.shape[1]:
        return im_ms

    # reshape image into vector and apply cloud mask
    vec = im_ms.reshape(im_ms.shape[0] * im_ms.shape[1], im_ms.shape[2])
    vec_mask = cloud_mask.reshape(im_ms.shape[0] * im_ms.shape[1])
    vec = vec[~vec_mask, :]
    # apply PCA to multispectral bands
    pca = decomposition.PCA()
    vec_pcs = pca.fit_transform(vec)

    # replace 1st PC with pan band (after matching histograms)
    vec_pan = im_pan.reshape(im_pan.shape[0] * im_pan.shape[1])
    vec_pan = vec_pan[~vec_mask]
    vec_pcs[:, 0] = hist_match(vec_pan, vec_pcs[:, 0])
    vec_ms_ps = pca.inverse_transform(vec_pcs)

    # reshape vector into image
    vec_ms_ps_full = np.ones((len(vec_mask), im_ms.shape[2])) * np.nan
    vec_ms_ps_full[~vec_mask, :] = vec_ms_ps
    im_ms_ps = vec_ms_ps_full.reshape(im_ms.shape[0], im_ms.shape[1], im_ms.shape[2])

    return im_ms_ps


def rescale_image_intensity(im, cloud_mask, prob_high):
    """
    Rescales the intensity of an image (multispectral or single band) by applying
    a cloud mask and clipping the prob_high upper percentile. This functions allows
    to stretch the contrast of an image, only for visualisation purposes.

    KV WRL 2018

    Arguments:
    -----------
    im: np.array
        Image to rescale, can be 3D (multispectral) or 2D (single band)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    prob_high: float
        probability of exceedence used to calculate the upper percentile

    Returns:
    -----------
    im_adj: np.array
        rescaled image
    """

    # lower percentile is set to 0
    prc_low = 0

    # reshape the 2D cloud mask into a 1D vector
    vec_mask = cloud_mask.reshape(im.shape[0] * im.shape[1])
    # if image contains several bands, stretch the contrast for each band
    if len(im.shape) > 2:
        # reshape into a vector
        vec = im.reshape(im.shape[0] * im.shape[1], im.shape[2])
        # # initiliase with NaN values
        vec_adj = np.ones((len(vec_mask), im.shape[2])) * np.nan
        # loop through the bands
        for i in range(im.shape[2]):
            # find the higher percentile (based on prob)
            prc_high = np.percentile(vec[~vec_mask, i], prob_high)
            # clip the image around the 2 percentiles and rescale the contrast
            vec_rescaled = exposure.rescale_intensity(
                vec[~vec_mask, i], in_range=(prc_low, prc_high)
            )
            vec_adj[~vec_mask, i] = vec_rescaled
        # reshape into image
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1], im.shape[2])

    # if image only has 1 bands (grayscale image)
    else:
        vec = im.reshape(im.shape[0] * im.shape[1])
        vec_adj = np.ones(len(vec_mask)) * np.nan
        prc_high = np.percentile(vec[~vec_mask], prob_high)
        vec_rescaled = exposure.rescale_intensity(
            vec[~vec_mask], in_range=(prc_low, prc_high)
        )
        vec_adj[~vec_mask] = vec_rescaled
        im_adj = vec_adj.reshape(im.shape[0], im.shape[1])

    return im_adj


def save_sar_image(
    im_ms: np.ndarray,
    date: str,
    satname: str,
    root_dir: str,
    quality: int = 100,
) -> None:
    """
    Saves a SAR image (e.g., Sentinel-1) as a titled JPEG figure.

    For a dual-polarization scene im_ms is the [VV, VH, VV-VH] composite and the
    preview is that composite as a false-colour RGB, one polarization per channel.
    A scene with a single polarization is shown as greyscale instead.

    Each band is stretched on its own 2nd-98th percentiles: the three bands cover
    very different dB ranges, so a shared stretch would saturate VV-VH.

    Args:
        im_ms (np.ndarray): 3D array with one band per composite band.
        date (str): Acquisition date string used in the filename.
        satname (str): Name of the satellite (e.g., 'S1').
        root_dir (str): Base directory where the image will be saved.
        quality (int, optional): unused, kept for call compatibility.

    Returns:
        None
    """
    if im_ms.ndim not in (2, 3):
        raise ValueError(f"Unexpected shape for SAR input: {im_ms.shape}")

    im_display, title = build_sar_preview(im_ms, date, satname)

    rgb_dir = os.path.join(root_dir, "RGB")
    os.makedirs(rgb_dir, exist_ok=True)

    plt.ioff()
    height, width = im_display.shape[:2]
    # keep the figure roughly the aspect ratio of the image, with room for the title
    fig, ax = plt.subplots(figsize=(8, 8 * height / max(width, 1) + 0.6))
    ax.imshow(im_display, cmap=None if im_display.ndim == 3 else "gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.savefig(
        os.path.join(rgb_dir, f"{date}_RGB_{satname}.jpg"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_sar_preview(im_ms, date, satname):
    """
    Builds the displayable SAR preview image and its title.

    Args:
        im_ms (np.ndarray): 2D band, or 3D array of composite/polarization bands.
        date (str): Acquisition date string.
        satname (str): Name of the satellite (e.g., 'S1').

    Returns:
        tuple: (np.ndarray to display, str title). The array is an (H, W, 3) RGB for
        a full composite, or (H, W) greyscale otherwise.
    """
    bands = SDS_tools.SAR_COMPOSITE_BANDS

    if im_ms.ndim == 3 and im_ms.shape[2] >= len(bands):
        im_display = np.dstack(
            [_stretch_sar_band(im_ms[:, :, k]) for k in range(len(bands))]
        )
        channels = ", ".join(
            f"{channel}={band}" for channel, band in zip("RGB", bands)
        )
        title = f"{date} {satname} - composite of the {'/'.join(bands)} bands ({channels})"
    else:
        band = im_ms if im_ms.ndim == 2 else im_ms[:, :, 0]
        im_display = _stretch_sar_band(band)
        title = f"{date} {satname}"

    return im_display, title


def _stretch_sar_band(band):
    """Rescales one SAR band to [0, 1] on its own 2nd-98th percentiles."""
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(band, [2, 98])
    if high <= low:
        return np.zeros_like(band, dtype="float64")
    return exposure.rescale_intensity(
        band.astype("float64"), in_range=(low, high), out_range=(0, 1)
    )


def save_multispectral_images(
    im_ms: np.ndarray,
    cloud_mask: np.ndarray,
    date: str,
    satname: str,
    root_dir: str,
    quality: int = 100,
) -> None:
    """
    Saves RGB, NIR, and SWIR1 bands from a multispectral image as JPEG files to the
    folders RGB/, NIR/, and SWIR/ under the root directory.

    Each band is extracted from a 3D image array, rescaled using percentile-based
    stretching (ignoring cloud-covered areas), converted to 8-bit, and saved in
    separate folders (RGB/, NIR/, SWIR/) under the root directory.

    Args:
        im_ms (np.ndarray): 3D array of multispectral bands ordered as [B, G, R, NIR, SWIR1].
        cloud_mask (np.ndarray): 2D boolean array where True indicates clouds.
        date (str): Acquisition date string used in the filename.
        satname (str): Name of the satellite (e.g., 'L5', 'S2').
        root_dir (str): Base directory where the images will be saved.
        quality (int, optional): JPEG quality level (default is 100).

    Returns:
        None
    """
    band_map = {
        "RGB": im_ms[:, :, [2, 1, 0]],
        "NIR": im_ms[:, :, 3],
        "SWIR": im_ms[:, :, 4],
    }

    for band_name, band_data in band_map.items():
        rescaled = rescale_image_intensity(band_data, cloud_mask, 99.9)
        rescaled_uint8 = img_as_ubyte(rescaled)

        band_dir = os.path.join(root_dir, band_name)
        os.makedirs(band_dir, exist_ok=True)

        filename = f"{date}_{band_name}_{satname}.jpg"
        imageio.imwrite(
            os.path.join(band_dir, filename), rescaled_uint8, quality=quality
        )


def create_jpg(
    im_ms: np.array,
    cloud_mask: np.array,
    date: str,
    satname: str,
    filepath: str,
    **kwargs,
) -> None:
    """
    Saves JPEG images for RGB, NIR, and SWIR bands from a multispectral image.
    For SAR images, it saves a single-band to the RGB folder.

    Originally created by KV WRL 2018 and modified in 2025

    Arguments:
    -----------
    im_ms: np.array
        3D array containing the pansharpened/down-sampled bands (B,G,R,NIR,SWIR1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    date: str
        string containing the date at which the image was acquired
    satname: str
        name of the satellite mission (e.g., 'L5')
        Available options are 'L5', 'L7', 'L8', 'S2', 'S1'
    filepath: str
        directory in which to save the images. Generally this is 'data/preprocessed'
    Kwargs:
        Currently not used but can be used to pass additional arguments

    Returns:
    -----------
        Saves a .jpg image corresponding to the preprocessed satellite image

    """
    if not os.path.isdir(filepath):
        os.makedirs(filepath, exist_ok=True)

    if satname == "S1":
        save_sar_image(im_ms, date, satname, filepath)
    else:
        save_multispectral_images(im_ms, cloud_mask, date, satname, filepath)


def save_single_jpg(
    filename,
    tif_paths,
    satname,
    sitename,
    cloud_thresh,
    cloud_mask_issue,
    filepath_data,
    collection,
    **kwargs,
):
    """
    Save a jpg for a single set of tifs

    KV WRL 2018

    Arguments:
    -----------
    cloud_thresh: float
        value between 0 and 1 indicating the maximum cloud fraction in
        the cropped image that is accepted
    cloud_mask_issue: boolean
        True if there is an issue with the cloud mask and sand pixels
        are erroneously being masked on the images

    Returns:
    -----------
    Creates RGB, NIR, SWIR as .jpg in:
    RGB saved under data/preprocessed/RGB
    NIR saved under data/preprocessed/NIR
    SWIR saved under data/preprocessed/SWIR
    """
    do_cloud_mask = kwargs.get("apply_cloud_mask", True)

    # Set up output directory
    jpg_directory = os.path.join(filepath_data, sitename, "jpg_files", "preprocessed")
    os.makedirs(jpg_directory, exist_ok=True)

    # get locations to each tif file for ms, pan, mask, swir (if applicable for satname)
    fn = SDS_tools.get_filenames(filename, tif_paths, satname)
    # preprocess the image and perform pansharpening
    settings = {
        "apply_cloud_mask": do_cloud_mask,
        "cloud_mask_issue": cloud_mask_issue,
        "pan_off": False,
        "s2cloudless_prob": kwargs.get("s2cloudless_prob", 60),
    }
    im_ms, georef, cloud_mask, im_nodata = preprocess_image(
        fn, satname, settings, collection
    )

    # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
    # compute cloud cover percentage (without no data pixels)
    cloud_cover = calculate_cloud_cover_excluding_nodata(cloud_mask, im_nodata)
    # cloud_cover = np.divide(
    #     sum(sum(cloud_mask_adv.astype(int))), (sum(sum((~im_nodata).astype(int))))
    # )

    if cloud_cover > cloud_thresh or cloud_cover == 1:
        return  # skip image if cloud cover is above threshold

    # save .jpg with date and satellite in the title
    date = filename[:19]
    plt.ioff()  # turning interactive plotting off
    create_jpg(
        im_ms, cloud_mask, date, satname, jpg_directory, im_nodata=im_nodata, **kwargs
    )


def save_jpg(metadata, settings, **kwargs):
    """
    Saves a .jpg image for all the images contained in metadata.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        's2cloudless_prob': float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask
        'use_matplotlib': boolean
            False to save a .jpg and True to save as matplotlib plots

    Returns:
    -----------
    Stores the images as .jpg in a folder named /preprocessed

    """

    sitename = settings["inputs"]["sitename"]
    cloud_thresh = settings["cloud_thresh"]
    s2cloudless_prob = settings.get("s2cloudless_prob", 60)
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]

    # create subfolder to store the jpg files
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "preprocessed")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)

    # loop through satellite list
    print("Saving images as jpg:")
    for satname in metadata.keys():
        filepath = SDS_tools.get_filepath(settings["inputs"], satname)
        filenames = metadata[satname]["filenames"]
        print("%s: %d images" % (satname, len(filenames)))
        # loop through images
        for i in range(len(filenames)):
            print("\r%d%%" % int((i + 1) / len(filenames) * 100), end="")
            # image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            # SAR has no cloud mask and no pansharpening, so it takes the simpler
            # preprocess_image path instead of preprocess_single
            if satname == "S1":
                im_ms, georef, cloud_mask, im_nodata = preprocess_image(
                    fn, satname, settings, collection
                )
                plt.ioff()  # turning interactive plotting off
                create_jpg(
                    im_ms,
                    cloud_mask,
                    filenames[i][:19],
                    satname,
                    filepath_jpg,
                    im_nodata=im_nodata,
                    **kwargs,
                )
                continue
            # read and preprocess image
            apply_cloud_mask = settings.get("apply_cloud_mask", True)
            try:
                im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = (
                    preprocess_single(
                        fn,
                        satname,
                        settings["cloud_mask_issue"],
                        settings["pan_off"],
                        collection,
                        apply_cloud_mask,
                        s2cloudless_prob=s2cloudless_prob,
                    )
                )
            except SkipImageError as exc:
                # Skip only the failing image and continue creating jpgs.
                print(f"{satname}: {exc}")
                continue

            # compute cloud_cover percentage (with no data pixels)
            cloud_cover_combined = np.divide(
                sum(sum(cloud_mask.astype(int))),
                (cloud_mask.shape[0] * cloud_mask.shape[1]),
            )
            if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
                continue

            # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
            cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
            # compute updated cloud cover percentage (without no data pixels)
            cloud_cover = np.divide(
                sum(sum(cloud_mask_adv.astype(int))),
                (sum(sum((~im_nodata).astype(int)))),
            )
            # skip image if cloud cover is above threshold
            if cloud_cover > cloud_thresh or cloud_cover == 1:
                continue
            # save .jpg with date and satellite in the title
            date = filenames[i][:19]
            plt.ioff()  # turning interactive plotting off
            create_jpg(
                im_ms,
                cloud_mask,
                date,
                satname,
                filepath_jpg,
                im_nodata=im_nodata,
                **kwargs,
            )
        print("")
    # print the location where the images have been saved
    print(
        "Satellite images saved as .jpg in "
        + os.path.join(filepath_data, sitename, "jpg_files", "preprocessed")
    )


def get_reference_sl(metadata, settings):
    """
    Allows the user to manually digitize a reference shoreline that is used seed
    the shoreline detection algorithm. The reference shoreline helps to detect
    the outliers, making the shoreline detection more robust.

    KV WRL 2018

    Arguments:
    -----------
    metadata: dict
        contains all the information about the satellite images that were downloaded
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'cloud_thresh': float
            value between 0 and 1 indicating the maximum cloud fraction in
            the cropped image that is accepted
        'cloud_mask_issue': boolean
            True if there is an issue with the cloud mask and sand pixels
            are erroneously being masked on the images
        'output_epsg': int
            output spatial reference system as EPSG code

    Returns:
    -----------
    reference_shoreline: np.array
        coordinates of the reference shoreline that was manually digitized.
        This is also saved as a .pkl and .geojson file.

    """

    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]
    pts_coords = []
    # check if reference shoreline already exists in the corresponding folder
    filepath = os.path.join(filepath_data, sitename)
    filename = sitename + "_reference_shoreline.pkl"
    # if it exist, load it and return it
    if filename in os.listdir(filepath):
        print("Reference shoreline already exists and was loaded")
        # load_pickle() rather than pickle.load(): reference shorelines saved by a
        # NumPy 2 environment name numpy._core, which NumPy 1.x cannot import.
        refsl = SDS_tools.load_pickle(
            os.path.join(filepath, sitename + "_reference_shoreline.pkl")
        )
        return refsl
    # otherwise get the user to manually digitise a shoreline on
    # S2, L8, L9 or L5 images (no L7 because of scan line error)
    # first try to use S2 images (10m res for manually digitizing the reference shoreline)
    if (
        "S2" in metadata.keys()
        and metadata.get("S2", {"filenames": []})["filenames"] != []
    ):
        satname = "S2"
    # if no S2 images, use L8 or L9 (15m res in the RGB with pansharpening)
    elif (
        "L8" in metadata.keys()
        and metadata.get("L8", {"filenames": []})["filenames"] != []
    ):
        satname = "L8"
    elif (
        "L9" in metadata.keys()
        and metadata.get("L9", {"filenames": []})["filenames"] != []
    ):
        satname = "L9"
    # if no S2, L8 or L9 use L5 (30m res)
    elif (
        "L5" in metadata.keys()
        and metadata.get("L5", {"filenames": []})["filenames"] != []
    ):
        satname = "L5"
    # if only L7 images, ask user to download other images
    else:
        raise Exception(
            "You cannot digitize the shoreline on L7 or S1 images (because of gaps in the images), add another L8, S2 or L5 to your dataset."
        )

    filepath = SDS_tools.get_filepath(settings["inputs"], satname)
    filenames = metadata[satname]["filenames"]
    # create figure
    fig, ax = plt.subplots(1, 1, figsize=[18, 9], tight_layout=True)
    mng = plt.get_current_fig_manager()
    # loop trhough the images
    for i in range(len(filenames)):
        # read image
        apply_cloud_mask = settings.get("apply_cloud_mask", True)
        fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
        try:
            im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = preprocess_single(
                fn,
                satname,
                settings["cloud_mask_issue"],
                settings["pan_off"],
                collection,
                apply_cloud_mask,
                settings.get("s2cloudless_prob", 60),
            )
        except SkipImageError as exc:
            # Skip only the failing image and continue scanning the reference set.
            print(f"{satname}: {exc}")
            continue

        # compute cloud_cover percentage (with no data pixels)
        cloud_cover_combined = np.divide(
            sum(sum(cloud_mask.astype(int))),
            (cloud_mask.shape[0] * cloud_mask.shape[1]),
        )
        if cloud_cover_combined > 0.99:  # if 99% of cloudy pixels in image skip
            continue
        # remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
        cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
        # compute updated cloud cover percentage (without no data pixels)
        cloud_cover = np.divide(
            sum(sum(cloud_mask_adv.astype(int))), (sum(sum((~im_nodata).astype(int))))
        )
        # skip image if cloud cover is above threshold
        if cloud_cover > settings["cloud_thresh"]:
            print(
                f"cloud cover {cloud_cover} is above threshold {settings['cloud_thresh']}"
            )
            continue
        # rescale image intensity for display purposes
        im_RGB = rescale_image_intensity(im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9)
        # plot the image RGB on a figure
        ax.axis("off")
        ax.imshow(im_RGB)
        # decide if the image if good enough for digitizing the shoreline
        ax.set_title(
            "Press <right arrow> if image is clear enough to digitize the shoreline.\n"
            + "If the image is cloudy press <left arrow> to get another image",
            fontsize=14,
        )
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
        skip_image = False
        key_event = {}

        def press(event):
            # store what key was pressed in the dictionary
            key_event["pressed"] = event.key

        # let the user press a key, right arrow to keep the image, left arrow to skip it
        # to break the loop the user can press 'escape'
        while True:
            btn_keep = plt.text(
                1.1,
                0.9,
                "keep ⇨",
                size=12,
                ha="right",
                va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            btn_skip = plt.text(
                -0.1,
                0.9,
                "⇦ skip",
                size=12,
                ha="left",
                va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            btn_esc = plt.text(
                0.5,
                0,
                "<esc> to quit",
                size=12,
                ha="center",
                va="top",
                transform=ax.transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            plt.draw()
            fig.canvas.mpl_connect("key_press_event", press)
            plt.waitforbuttonpress()
            # after button is pressed, remove the buttons
            btn_skip.remove()
            btn_keep.remove()
            btn_esc.remove()
            # keep/skip image according to the pressed key, 'escape' to break the loop
            if key_event.get("pressed") == "right":
                skip_image = False
                break
            elif key_event.get("pressed") == "left":
                skip_image = True
                break
            elif key_event.get("pressed") == "escape":
                plt.close()
                raise StopIteration("User cancelled checking shoreline detection")
            else:
                plt.waitforbuttonpress()

        if skip_image:
            ax.clear()
            continue
        else:
            # create two new buttons
            add_button = plt.text(
                0,
                0.9,
                "add",
                size=16,
                ha="left",
                va="top",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            end_button = plt.text(
                1,
                0.9,
                "end",
                size=16,
                ha="right",
                va="top",
                transform=plt.gca().transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            # add multiple reference shorelines (until user clicks on <end> button)
            pts_sl = np.expand_dims(np.array([np.nan, np.nan]), axis=0)
            geoms = []
            while 1:
                add_button.set_visible(False)
                end_button.set_visible(False)
                # update title (instructions)
                ax.set_title(
                    "Click points along the shoreline (enough points to capture the beach curvature).\n"
                    + "Start at one end of the beach.\n"
                    + "When finished digitizing, click <ENTER>",
                    fontsize=14,
                )
                plt.draw()
                # let user click on the shoreline
                pts = ginput(n=50000, timeout=-1, show_clicks=True)
                pts_pix = np.array(pts)
                # convert pixel coordinates to world coordinates
                pts_world = SDS_tools.convert_pix2world(pts_pix[:, [1, 0]], georef)
                # interpolate between points clicked by the user (1m resolution)
                pts_world_interp = np.expand_dims(np.array([np.nan, np.nan]), axis=0)
                for k in range(len(pts_world) - 1):
                    pt_dist = np.linalg.norm(pts_world[k, :] - pts_world[k + 1, :])
                    xvals = np.arange(0, pt_dist)
                    yvals = np.zeros(len(xvals))
                    pt_coords = np.zeros((len(xvals), 2))
                    pt_coords[:, 0] = xvals
                    pt_coords[:, 1] = yvals
                    phi = 0
                    deltax = pts_world[k + 1, 0] - pts_world[k, 0]
                    deltay = pts_world[k + 1, 1] - pts_world[k, 1]
                    phi = np.pi / 2 - np.arctan2(deltax, deltay)
                    tf = transform.EuclideanTransform(
                        rotation=phi, translation=pts_world[k, :]
                    )
                    pts_world_interp = np.append(
                        pts_world_interp, tf(pt_coords), axis=0
                    )
                pts_world_interp = np.delete(pts_world_interp, 0, axis=0)
                # save as geometry (to create .geojson file later)
                geoms.append(geometry.LineString(pts_world_interp))
                # convert to pixel coordinates and plot
                pts_pix_interp = SDS_tools.convert_world2pix(pts_world_interp, georef)
                pts_sl = np.append(pts_sl, pts_world_interp, axis=0)
                ax.plot(pts_pix_interp[:, 0], pts_pix_interp[:, 1], "r--")
                ax.plot(pts_pix_interp[0, 0], pts_pix_interp[0, 1], "ko")
                ax.plot(pts_pix_interp[-1, 0], pts_pix_interp[-1, 1], "ko")
                # update title and buttons
                add_button.set_visible(True)
                end_button.set_visible(True)
                ax.set_title(
                    "click on <add> to digitize another shoreline or on <end> to finish and save the shoreline(s)",
                    fontsize=14,
                )
                plt.draw()
                # let the user click again (<add> another shoreline or <end>)
                pt_input = ginput(n=1, timeout=-1, show_clicks=False)
                pt_input = np.array(pt_input)
                # if user clicks on <end>, save the points and break the loop
                if pt_input[0][0] > im_ms.shape[1] / 2:
                    add_button.set_visible(False)
                    end_button.set_visible(False)
                    plt.title(
                        "Reference shoreline saved as "
                        + sitename
                        + "_reference_shoreline.pkl and "
                        + sitename
                        + "_reference_shoreline.geojson"
                    )
                    plt.draw()
                    ginput(n=1, timeout=3, show_clicks=False)
                    plt.close()
                    break
            pts_sl = np.delete(pts_sl, 0, axis=0)
            # convert world image coordinates to user-defined coordinate system
            image_epsg = metadata[satname]["epsg"][i]
            pts_coords = SDS_tools.convert_epsg(
                pts_sl, image_epsg, settings["output_epsg"]
            )
            # save the reference shoreline as .pkl
            filepath = os.path.join(filepath_data, sitename)
            # save_path = os.path.join(filepath, sitename + "_reference_shoreline.json")
            # write_to_json(save_path, pts_coords)
            with open(
                os.path.join(filepath, sitename + "_reference_shoreline.pkl"), "wb"
            ) as f:
                pickle.dump(pts_coords, f)

            # also store as .geojson in case user wants to drag-and-drop on GIS for verification
            # for k,line in enumerate(geoms):
            #     gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(line))
            #     gdf.index = [k]
            #     gdf.loc[k,'name'] = 'reference shoreline ' + str(k+1)
            #     # store into geodataframe
            #     if k == 0:
            #         gdf_all = gdf
            #     else:
            #         gdf_all = gdf_all.append(gdf)

            # initialize an empty list to store the individual GeoDataFrames
            gdf_list = []

            for k, line in enumerate(geoms):
                gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(line))
                gdf.index = [k]
                gdf.loc[k, "name"] = "reference shoreline " + str(k + 1)
                gdf_list.append(gdf)

            # concatenate all GeoDataFrames in the list into a single GeoDataFrame
            gdf_all = pd.concat(gdf_list, ignore_index=True)

            gdf_all.crs = f"EPSG:{image_epsg}"
            # convert from image_epsg to user-defined coordinate system
            gdf_all = gdf_all.to_crs(f"EPSG:{settings['output_epsg']}")
            # save as geojson
            gdf_all.to_file(
                os.path.join(filepath, sitename + "_reference_shoreline.geojson"),
                driver="GeoJSON",
                encoding="utf-8",
            )
            print("Reference shoreline has been saved in " + filepath)
            break

    # check if a shoreline was digitised
    if len(pts_coords) == 0:
        raise Exception(
            "No cloud free images are available to digitise the reference shoreline,"
            + "download more images and try again"
        )

    return pts_coords
