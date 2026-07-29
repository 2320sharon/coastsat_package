"""
This module contains utilities to work with satellite images

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# Standard library imports
import os
import re
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional

import numpy as np
import bisect
import pickle

# Third-party imports
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from astropy.convolution import convolve
from osgeo import gdal, osr
from scipy import stats, interpolate
from shapely import geometry
import skimage.morphology as morphology
import skimage.transform as transform
import pytz
import pdb
import scipy.ndimage
from PIL import Image

# Canonical order in which Sentinel-1 polarizations are stored on disk and stacked into im_ms.
# Every place that turns a list of polarizations into folders or image channels uses this order,
# so the channel order never depends on the order the user happened to list them in.
SAR_POLARIZATIONS = ("VV", "VH", "HH", "HV")

# Polarizations downloaded when the user does not ask for a specific set. Both are
# needed for the [VV, VH, VV-VH] composite that the shoreline detection works on.
DEFAULT_SAR_POLARIZATIONS = ("VV", "VH")

# Bands of the Sentinel-1 composite, built in memory and never written to disk.
# VV-VH is a dB difference (the linear VV/VH ratio), which suppresses the surface
# roughness common to both polarizations.
SAR_COMPOSITE_BANDS = ("VV", "VH", "VV-VH")

# Matches the polarization token in a downloaded SAR filename. The token is not always last:
# SDS_download.get_file_name() produces '{date}_{sat}_{site}_{polar}_dup{N}.tif' for duplicates.
_POL_RE = re.compile(
    r"_(%s)(?=(?:_dup\d+)?\.tif$)" % "|".join(SAR_POLARIZATIONS), re.IGNORECASE
)

# np.seterr(all='ignore') # raise/ignore divisions by 0 and nans

###################################################################################################
# SCIKIT-IMAGE COMPATIBILITY
###################################################################################################

# scikit-image is retiring two spellings we rely on, on different schedules:
# morphology.square() is deprecated since 0.25 and goes in 0.27, and the
# morphology.binary_* operators are deprecated since 0.26 and go in 0.28. Their
# replacements are not available on older installs -- footprint_rectangle() only
# exists from 0.25. So resolve the spelling against whatever is
# installed rather than hard-coding either one.
#
# Switching binary_dilation()/binary_opening() to dilation()/opening() is only safe
# because every footprint we pass (square, disk) is symmetric under 180-degree
# rotation; the deprecation notice warns that the replacements skip the footprint
# mirroring, which changes results for non-symmetric footprints only. Below 0.25 we
# keep calling the original binary_* functions, so behaviour on old installs is
# untouched rather than merely believed-equivalent.
#
# One flag drives both swaps, so the operators switch at 0.25 rather than at 0.26
# when they were actually deprecated. That is deliberate -- it keeps a single
# version boundary instead of two -- and is safe because dilation()/opening() have
# accepted boolean input and matched the binary_* results for symmetric footprints
# for far longer than either deprecation.
_HAS_FOOTPRINT_RECTANGLE = hasattr(morphology, "footprint_rectangle")


def footprint_square(width: int) -> np.ndarray:
    """Square footprint of side `width`, spelled for the installed scikit-image."""
    if _HAS_FOOTPRINT_RECTANGLE:
        return morphology.footprint_rectangle((width, width))
    return morphology.square(width)


def binary_opening(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Binary morphological opening, spelled for the installed scikit-image."""
    if _HAS_FOOTPRINT_RECTANGLE:
        return morphology.opening(image, footprint)
    return morphology.binary_opening(image, footprint)


def binary_dilation(image: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    """Binary morphological dilation, spelled for the installed scikit-image."""
    if _HAS_FOOTPRINT_RECTANGLE:
        return morphology.dilation(image, footprint)
    return morphology.binary_dilation(image, footprint)


###################################################################################################
# NUMPY COMPATIBILITY
###################################################################################################

# NumPy 2 renamed the private numpy.core package to numpy._core, and that name is
# baked into every .pkl written under NumPy 2. Loading one on NumPy 1.x fails with
# "ModuleNotFoundError: No module named 'numpy._core...'". NumPy 2 ships a
# numpy.core shim so the opposite direction already works, which is why only this
# one direction needs bridging.
#
# The remap is attempted only after the real import fails, so on NumPy 2 -- where
# numpy._core genuinely exists -- this behaves exactly like pickle.Unpickler and
# nothing changes. Everything used here (Unpickler.find_class) is stdlib and
# unchanged across every supported Python version.


class _NumpyCompatUnpickler(pickle.Unpickler):
    """Unpickler that can also read NumPy-2-written pickles on NumPy 1.x."""

    def find_class(self, module: str, name: str):
        # exact package or a submodule of it -- not merely a "numpy._core" prefix
        if module == "numpy._core" or module.startswith("numpy._core."):
            try:
                return super().find_class(module, name)
            except (ImportError, AttributeError):
                # numpy._core.numeric -> numpy.core.numeric
                legacy = "numpy.core" + module[len("numpy._core") :]
                return super().find_class(legacy, name)
        return super().find_class(module, name)


def load_pickle(filepath: str):
    """pickle.load() for a file path, tolerant of the NumPy 1/2 module rename."""
    with open(filepath, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


###################################################################################################
# COORDINATES CONVERSION FUNCTIONS
###################################################################################################


def convert_pix2world(points: np.array, georef: np.array):
    """
    Converts pixel coordinates (pixel row and column) to world projected
    coordinates performing an affine transformation.

    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (row first and column second)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]

    Returns:
    -----------
    points_converted: np.array or list of np.array
        converted coordinates, first columns with X and second column with Y

    """

    # make affine transformation matrix
    aff_mat = np.array(
        [
            [georef[1], georef[2], georef[0]],
            [georef[4], georef[5], georef[3]],
            [0, 0, 1],
        ]
    )
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points):
            tmp = arr[:, [1, 0]]
            points_converted.append(tform(tmp))

    # if single array
    elif type(points) is np.ndarray:
        tmp = points[:, [1, 0]]
        points_converted = tform(tmp)

    else:
        raise Exception("invalid input type")

    return points_converted


def convert_world2pix(points, georef):
    """
    Converts world projected coordinates (X,Y) to image coordinates
    (pixel row and column) performing an affine transformation.

    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.array
        array with 2 columns (X,Y)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]

    Returns:
    -----------
    points_converted: np.array or list of np.array
        converted coordinates (pixel row and column)

    """

    # make affine transformation matrix
    aff_mat = np.array(
        [
            [georef[1], georef[2], georef[0]],
            [georef[4], georef[5], georef[3]],
            [0, 0, 1],
        ]
    )
    # create affine transformation
    tform = transform.AffineTransform(aff_mat)

    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points):
            points_converted.append(tform.inverse(points))

    # if single array
    elif type(points) is np.ndarray:
        points_converted = tform.inverse(points)

    else:
        print("invalid input type")
        raise

    return points_converted


def convert_epsg(points, epsg_in, epsg_out):
    """
    Converts from one spatial reference to another using the epsg codes

    KV WRL 2018

    Arguments:
    -----------
    points: np.array or list of np.ndarray
        array with 2 columns (rows first and columns second)
    epsg_in: int
        epsg code of the spatial reference in which the input is
    epsg_out: int
        epsg code of the spatial reference in which the output will be

    Returns:
    -----------
    points_converted: np.array or list of np.array
        converted coordinates from epsg_in to epsg_out

    """

    # define input and output spatial references
    inSpatialRef = osr.SpatialReference()
    inSpatialRef.ImportFromEPSG(epsg_in)
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(epsg_out)
    # create a coordinates transform
    coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
    # if list of arrays
    if type(points) is list:
        points_converted = []
        # iterate over the list
        for i, arr in enumerate(points):
            points_converted.append(np.array(coordTransform.TransformPoints(arr)))
    # if single array
    elif type(points) is np.ndarray:
        points_converted = np.array(coordTransform.TransformPoints(points))
    else:
        raise Exception("invalid input type")

    return points_converted


###################################################################################################
# IMAGE ANALYSIS FUNCTIONS
###################################################################################################


def nd_index(im1, im2, cloud_mask):
    """
    Computes normalised difference index on 2 images (2D), given a cloud mask (2D).

    KV WRL 2018

    Arguments:
    -----------
    im1: np.array
        first image (2D) with which to calculate the ND index
    im2: np.array
        second image (2D) with which to calculate the ND index
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are

    Returns:
    -----------
    im_nd: np.array
        Image (2D) containing the ND index

    """

    # reshape the cloud mask
    vec_mask = cloud_mask.reshape(im1.shape[0] * im1.shape[1])
    # initialise with NaNs
    vec_nd = np.ones(len(vec_mask)) * np.nan
    # reshape the two images
    vec1 = im1.reshape(im1.shape[0] * im1.shape[1])
    vec2 = im2.reshape(im2.shape[0] * im2.shape[1])
    # compute the normalised difference index
    temp = np.divide(
        vec1[~vec_mask] - vec2[~vec_mask], vec1[~vec_mask] + vec2[~vec_mask]
    )
    vec_nd[~vec_mask] = temp
    # reshape into image
    im_nd = vec_nd.reshape(im1.shape[0], im1.shape[1])

    return im_nd


def image_std(image, radius):
    """
    Calculates the standard deviation of an image, using a moving window of
    specified radius. Uses astropy's convolution library'

    Arguments:
    -----------
    image: np.array
        2D array containing the pixel intensities of a single-band image
    radius: int
        radius defining the moving window used to calculate the standard deviation.
        For example, radius = 1 will produce a 3x3 moving window.

    Returns:
    -----------
    win_std: np.array
        2D array containing the standard deviation of the image

    """

    # convert to float
    image = image.astype(float)
    # first pad the image
    image_padded = np.pad(image, radius, "reflect")
    # window size
    win_rows, win_cols = radius * 2 + 1, radius * 2 + 1
    # calculate std with uniform filters
    win_mean = convolve(
        image_padded,
        np.ones((win_rows, win_cols)),
        boundary="extend",
        normalize_kernel=True,
        nan_treatment="interpolate",
        preserve_nan=True,
    )
    win_sqr_mean = convolve(
        image_padded**2,
        np.ones((win_rows, win_cols)),
        boundary="extend",
        normalize_kernel=True,
        nan_treatment="interpolate",
        preserve_nan=True,
    )
    win_var = win_sqr_mean - win_mean**2
    win_std = np.sqrt(win_var)
    # remove padding
    win_std = win_std[radius:-radius, radius:-radius]

    return win_std


def mask_raster(fn, mask):
    """
    Masks a .tif raster using GDAL.

    Arguments:
    -----------
    fn: str
        filepath + filename of the .tif raster
    mask: np.array
        array of boolean where True indicates the pixels that are to be masked

    Returns:
    -----------
    Overwrites the .tif file directly

    """

    # open raster
    raster = gdal.Open(fn, gdal.GA_Update)
    # mask raster
    for i in range(raster.RasterCount):
        out_band = raster.GetRasterBand(i + 1)
        out_data = out_band.ReadAsArray()
        out_band.SetNoDataValue(0)
        no_data_value = out_band.GetNoDataValue()
        out_data[mask] = no_data_value
        out_band.WriteArray(out_data)
    # close dataset and flush cache
    raster = None


def get_image_bounds(fn):
    """
    Returns a polygon with the bounds of the image in the .tif file

    KV WRL 2020

    Arguments:
    -----------
    fn: str
        path to the image (.tif file)

    Returns:
    -----------
    bounds_polygon: shapely.geometry.Polygon
        polygon with the image bounds

    """

    # nested functions to get the extent
    # copied from https://gis.stackexchange.com/questions/57834/how-to-get-raster-corner-coordinates-using-python-gdal-bindings
    def GetExtent(gt, cols, rows):
        "Return list of corner coordinates from a geotransform"
        ext = []
        xarr = [0, cols]
        yarr = [0, rows]
        for px in xarr:
            for py in yarr:
                x = gt[0] + (px * gt[1]) + (py * gt[2])
                y = gt[3] + (px * gt[4]) + (py * gt[5])
                ext.append([x, y])
            yarr.reverse()
        return ext

    # load .tif file and get bounds
    if not os.path.exists(fn):
        raise FileNotFoundError(f"{fn}")
    data = gdal.Open(fn, gdal.GA_ReadOnly)
    # Check if data is null meaning the open failed
    if data is None:
        print("TIF file: ", fn, "cannot be opened")
        os.remove(fn)
        raise AttributeError
    else:
        gt = data.GetGeoTransform()
        cols = data.RasterXSize
        rows = data.RasterYSize
        ext = GetExtent(gt, cols, rows)

    return geometry.Polygon(ext)


def get_image_dimensions(image_path):
    "function to get image dimensions with GDAL"
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception("Failed to open the image file %s" % image_path)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    dataset = None

    return width, height


###################################################################################################
# UTILITIES
###################################################################################################


def get_polarization_from_filename(filename: str) -> Optional[str]:
    """
    Returns the polarization token carried by a SAR filename, or None if it has none.

    Arguments:
    -----------
    filename: str
        name of a downloaded SAR image, e.g. '2023-12-03-19-15-49_S1_site_VH.tif'

    Returns:
    -----------
    str or None
        the polarization ('VV', 'VH', 'HH' or 'HV'), or None if the name has no token

    """
    match = _POL_RE.search(os.path.basename(filename))
    return match.group(1).upper() if match else None


def swap_polarization_in_filename(filename: str, polarization: str) -> str:
    """
    Returns the filename with its polarization token replaced by `polarization`.

    Used to find the sibling files of the same scene, which live in a folder per
    polarization but otherwise share a name.

    Arguments:
    -----------
    filename: str
        name of a downloaded SAR image
    polarization: str
        polarization to substitute in

    Returns:
    -----------
    str
        the filename with the new polarization token, unchanged if it had none

    """
    if not _POL_RE.search(os.path.basename(filename)):
        return filename
    return _POL_RE.sub(f"_{polarization}", filename, count=1)


# How a shoreline was derived, recorded per scene in output['segmentation_method'].
# It says what output['MNDWI_threshold'] holds for that scene, which is the only way to
# know whether a threshold-range filter can be applied to it - see
# SDS_transects.threshold_filter_applies.
SEGMENTATION_MNDWI = "mndwi"  # optical: MNDWI contour, threshold is an MNDWI value
SEGMENTATION_SAR_OTSU = "sar_otsu"  # SAR: Otsu threshold, in dB
SEGMENTATION_SAR_MODEL = "sar_model"  # SAR: ONNX segmentation, no threshold (NaN)


def build_sar_composite(im_polarizations, polarizations):
    """
    Builds the [VV, VH, VV-VH] composite from a stack of polarization bands.

    The third band is the dB difference of the first two. Returns None when the
    scene does not carry both VV and VH, which is the case for images downloaded
    before multi-polarization support.

    Arguments:
    -----------
    im_polarizations: np.ndarray
        (H, W, n) stack of polarization bands, as returned by read_sar_image
    polarizations: list of str
        polarization of each channel, in the same order

    Returns:
    -----------
    np.ndarray or None
        (H, W, 3) composite with bands SAR_COMPOSITE_BANDS, or None if VV and VH
        are not both available

    """
    polarizations = list(polarizations)
    if "VV" not in polarizations or "VH" not in polarizations:
        return None

    im_vv = im_polarizations[:, :, polarizations.index("VV")]
    im_vh = im_polarizations[:, :, polarizations.index("VH")]

    return np.stack([im_vv, im_vh, im_vv - im_vh], axis=2)


def create_folder_structure(im_folder, satname, polars: Optional[str] = None):
    """
    Create the structure of subfolders for each satellite mission

    KV WRL 2018

    Arguments:
    -----------
    im_folder: str
        folder where the images are to be downloaded
    satname:
        name of the satellite mission
    polars: str or list of str, optional
        Sentinel-1 polarizations to create a folder for, e.g. ['VV','VH'].
        Ignored for the optical missions. Defaults to DEFAULT_SAR_POLARIZATIONS.

    Returns:
    -----------
    filepaths: list of str
        filepaths of the folders that were created. The first entry is always the
        metadata folder; for S1 the rest are one folder per polarization, in
        SAR_POLARIZATIONS order.
    """

    # one folder for the metadata (common to all satellites)
    filepaths = [os.path.join(im_folder, satname, "meta")]
    # subfolders depending on satellite mission
    if satname == "L5":
        filepaths.append(os.path.join(im_folder, satname, "ms"))
        filepaths.append(os.path.join(im_folder, satname, "mask"))
    elif satname in ["L7", "L8", "L9"]:
        filepaths.append(os.path.join(im_folder, satname, "ms"))
        filepaths.append(os.path.join(im_folder, satname, "pan"))
        filepaths.append(os.path.join(im_folder, satname, "mask"))
    elif satname in ["S2"]:
        filepaths.append(os.path.join(im_folder, satname, "ms"))
        filepaths.append(os.path.join(im_folder, satname, "swir"))
        filepaths.append(os.path.join(im_folder, satname, "mask"))
    elif satname in ["S1"]:
        # one folder per polarization, each holding one single-band .tif per scene
        if not polars:
            polars = list(DEFAULT_SAR_POLARIZATIONS)
        if isinstance(polars, str):
            polars = [polars]
        requested = {
            str(polar).strip().upper() for polar in polars if str(polar).strip()
        }
        for polar in SAR_POLARIZATIONS:
            if polar in requested:
                filepaths.append(os.path.join(im_folder, satname, polar))
    # create the subfolders if they don't exist already
    for fp in filepaths:
        os.makedirs(fp, exist_ok=True)

    return filepaths


def read_sar_image(file_paths):
    """
    Reads the SAR image file(s) of one scene and stacks them into a single array.

    Each polarization of a scene is stored as its own single-band .tif in its own
    folder, so this takes one path per polarization and returns them stacked. The
    channel order is the order of `file_paths`, which get_filenames() produces in
    SAR_POLARIZATIONS order.

    Parameters:
        file_paths (str or list of str): Path(s) to the SAR image file(s) of one scene.

    Returns:
        im (np.ndarray): The image with one channel per polarization, shape (H, W, n).
        georef (np.ndarray): The georeference transformation array.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if len(file_paths) == 0:
        raise ValueError("No SAR image files were provided.")

    arrays = []
    georef = None
    ref_shape = None
    ref_path = None
    for file_path in file_paths:
        # Open the file using GDAL
        data = gdal.Open(file_path, gdal.GA_ReadOnly)
        if data is None:
            raise IOError(f"Could not open file: {file_path}")

        # each polarization file holds a single band
        band = data.GetRasterBand(1).ReadAsArray()
        geotransform = np.array(data.GetGeoTransform())

        if georef is None:
            georef, ref_shape, ref_path = geotransform, band.shape, file_path
        elif band.shape != ref_shape or not np.allclose(geotransform, georef):
            # the polarizations of one scene must share a grid or the stack is meaningless
            raise ValueError(
                "SAR polarization files are not on the same grid: "
                f"'{ref_path}' {ref_shape} vs '{file_path}' {band.shape}"
            )
        arrays.append(band)

    im = np.stack(arrays, axis=2)

    return im, georef


def read_sar_valid_mask(file_paths):
    """
    Reads the per-pixel validity mask of one SAR scene.

    A pixel is valid where every polarization file of the scene both reports it as valid
    (GDAL mask band, which honours the raster's nodata value) and holds a finite value.
    This is what the segmentation model uses to decide which pixels to neutralise on
    input and to blank out of the prediction, so that blank areas of a partially covered
    scene never produce a land or water call.

    Note the downloaded Sentinel-1 rasters carry nodata = 0.0, and 0 dB is a physically
    meaningful backscatter value, so a genuinely 0.0 dB pixel is treated as invalid.
    That is what the model's inference guide specifies and it did not occur in any of
    the scenes checked, but it is a real edge case.

    Call read_sar_image() first: it validates that the polarizations share a grid, which
    this function assumes.

    Parameters:
        file_paths (str or list of str): Path(s) to the SAR image file(s) of one scene.

    Returns:
        valid (np.ndarray): (H, W) boolean, True where every polarization is usable.
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    if len(file_paths) == 0:
        raise ValueError("No SAR image files were provided.")

    valid = None
    for file_path in file_paths:
        data = gdal.Open(file_path, gdal.GA_ReadOnly)
        if data is None:
            raise IOError(f"Could not open file: {file_path}")

        band = data.GetRasterBand(1)
        # the mask band is all 255 when the raster declares no nodata value
        band_valid = band.GetMaskBand().ReadAsArray() > 0
        band_valid &= np.isfinite(band.ReadAsArray())

        valid = band_valid if valid is None else (valid & band_valid)

    return valid


def get_filepath(inputs, satname):
    """
    Create filepath to the different folders containing the satellite images.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict with the following keys
        'sitename': str
            name of the site
        'polygon': list
            polygon containing the lon/lat coordinates to be extracted,
            longitudes in the first column and latitudes in the second column,
            there are 5 pairs of lat/lon with the fifth point equal to the first point:
            ```
            polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
            [151.3, -33.7]]]
            ```
        'dates': list of str
            list that contains 2 strings with the initial and final dates in
            format 'yyyy-mm-dd':
            ```
            dates = ['1987-01-01', '2018-01-01']
            ```
        'sat_list': list of str
            list that contains the names of the satellite missions to include:
            ```
            sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']
            ```
        'filepath': str
            filepath to the directory where the images are downloaded
    satname: str
        short name of the satellite mission ('L5','L7','L8','S2')

    Returns:
    -----------
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images

    """

    sitename = inputs["sitename"]
    filepath_data = inputs["filepath"]
    # access the images
    if satname == "L5":
        # access downloaded Landsat 5 images
        fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_mask]
    elif satname in ["L7", "L8", "L9"]:
        # access downloaded Landsat 7 images
        fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
        fp_pan = os.path.join(filepath_data, sitename, satname, "pan")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_pan, fp_mask]
    elif satname == "S2":
        # access downloaded Sentinel 2 images
        fp_ms = os.path.join(filepath_data, sitename, satname, "ms")
        fp_swir = os.path.join(filepath_data, sitename, satname, "swir")
        fp_mask = os.path.join(filepath_data, sitename, satname, "mask")
        filepath = [fp_ms, fp_swir, fp_mask]
    elif satname == "S1":
        # access downloaded Sentinel 1 images: one folder per polarization.
        # Only the folders that exist are returned, so a session downloaded before
        # multi-polarization support (VH only) still resolves.
        s1_root = os.path.join(filepath_data, sitename, satname)
        filepath = [
            os.path.join(s1_root, polar)
            for polar in SAR_POLARIZATIONS
            if os.path.isdir(os.path.join(s1_root, polar))
        ]

    return filepath


def get_filenames(filename, filepath, satname):
    """
    Creates filepath + filename for all the bands belonging to the same image.

    KV WRL 2018

    Arguments:
    -----------
    filename: str
        name of the downloaded satellite image as found in the metadata
    filepath: str or list of str
        contains the filepath(s) to the folder(s) containing the satellite images.
        A three part list containing the filepaths to the different bands of the image.
        For Landsat 5, it would be [fp_ms, fp_mask]
        For Landsat 7, 8 and 9, it would be [fp_ms, fp_pan, fp_mask]
        For Sentinel 2, it would be [fp_ms, fp_swir, fp_mask]
        For Sentinel 1, it would be one folder per polarization, e.g. [fp_VV, fp_VH]
    satname: str
        short name of the satellite mission

    Returns:
    -----------
    fn: str or list of str
        contains the filepath + filenames to access the satellite image.
        For Sentinel 1 this is one path per polarization present on disk, in
        SAR_POLARIZATIONS order, which is the order read_sar_image() stacks them.

    """

    if satname == "L5":
        fn_mask = filename.replace("ms.tif", "mask.tif")
        fn = [os.path.join(filepath[0], filename), os.path.join(filepath[1], fn_mask)]
    if satname in ["L7", "L8", "L9"]:
        fn_pan = filename.replace("ms.tif", "pan.tif")
        fn_mask = filename.replace("ms.tif", "mask.tif")
        fn = [
            os.path.join(filepath[0], filename),
            os.path.join(filepath[1], fn_pan),
            os.path.join(filepath[2], fn_mask),
        ]
    if satname == "S2":
        fn_swir = filename.replace("_ms", "_swir")
        fn_mask = filename.replace("_ms", "_mask")
        fn = [
            os.path.join(filepath[0], filename),
            os.path.join(filepath[1], fn_swir),
            os.path.join(filepath[2], fn_mask),
        ]
    if satname == "S1":
        # each polarization of the scene lives in its own folder but shares a name
        # apart from the polarization token, so derive the siblings from filename.
        # Only the files that exist are returned: a scene downloaded before
        # multi-polarization support has VH only, and may sit next to newer VV+VH scenes.
        candidates = filepath if isinstance(filepath, (list, tuple)) else [filepath]
        fn = []
        for folder in candidates:
            polar = os.path.basename(folder)
            path = os.path.join(folder, swap_polarization_in_filename(filename, polar))
            if os.path.exists(path):
                fn.append(path)
        if not fn:
            raise FileNotFoundError(
                f"No Sentinel-1 image found for '{filename}' in {list(candidates)}"
            )

    return fn


def merge_output(output):
    """
    Function to merge the output dictionnary, which has one key per satellite mission
    into a dictionnary containing all the shorelines and dates ordered chronologically.

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates, organised by
        satellite mission

    Returns:
    -----------
    output_all: dict
        contains the extracted shorelines in a single list sorted by date

    """

    # initialize output dict
    output_all = dict([])
    satnames = list(output.keys())
    for key in output[satnames[0]].keys():
        output_all[key] = []
    # create extra key for the satellite name
    output_all["satname"] = []
    # fill the output dict
    for satname in list(output.keys()):
        for key in output[satnames[0]].keys():
            output_all[key] = output_all[key] + output[satname][key]
        output_all["satname"] = output_all["satname"] + [
            _ for _ in np.tile(satname, len(output[satname]["dates"]))
        ]
    # sort chronologically
    idx_sorted = sorted(
        range(len(output_all["dates"])), key=output_all["dates"].__getitem__
    )
    for key in output_all.keys():
        output_all[key] = [output_all[key][i] for i in idx_sorted]

    return output_all


def remove_duplicates(output):
    """
    Function to remove from the output dictionnary entries containing shorelines for
    the same date and satellite mission. This happens when there is an overlap
    between adjacent satellite images.

    KV WRL 2020

    Arguments:
    -----------
        output: dict
            contains output dict with shoreline and metadata

    Returns:
    -----------
        output_no_duplicates: dict
            contains the updated dict where duplicates have been removed

    """
    # remove duplicates
    dates = output["dates"].copy()
    # find the pairs of images that are within 5 minutes of each other
    time_delta = 5 * 60  # 5 minutes in seconds
    pairs = []
    for i, date in enumerate(dates):
        # dummy value so it does not match it again
        dates[i] = pytz.utc.localize(datetime(1, 1, 1) + timedelta(days=i + 1))
        # calculate time difference
        time_diff = np.array([np.abs((date - _).total_seconds()) for _ in dates])
        # find the matching times and add to pairs list
        boolvec = time_diff <= time_delta
        if np.sum(boolvec) == 0:
            continue
        else:
            idx_dup = np.where(boolvec)[0][0]
            pairs.append([i, idx_dup])

    # if there are duplicates, only keep the longest shoreline
    if len(pairs) > 0:
        # initialise variables
        output_no_duplicates = dict([])
        idx_remove = []
        # for each pair
        for pair in pairs:
            # check if any of the shorelines are empty
            empty_bool = [(len(output["shorelines"][_]) < 2) for _ in pair]
            if np.all(empty_bool):  # if both empty remove both
                idx_remove.append(pair[0])
                idx_remove.append(pair[1])
            elif np.any(empty_bool):  # if one empty remove that one
                idx_remove.append(pair[np.where(empty_bool)[0][0]])
            else:  # remove the shorter shoreline and keep the longer one
                satnames = [output["satname"][_] for _ in pair]
                # keep Landsat 9 if it duplicates Landsat 7
                if "L9" in satnames and "L7" in satnames:
                    idx_remove.append(
                        pair[np.where([_ == "L7" for _ in satnames])[0][0]]
                    )
                else:  # keep the longest shorelines
                    sl0 = geometry.LineString(output["shorelines"][pair[0]])
                    sl1 = geometry.LineString(output["shorelines"][pair[1]])
                    if sl0.length >= sl1.length:
                        idx_remove.append(pair[1])
                    else:
                        idx_remove.append(pair[0])
        # create a new output structure with all the duplicates removed
        idx_remove = sorted(idx_remove)
        idx_all = np.linspace(0, len(dates) - 1, len(dates)).astype(int)
        idx_keep = list(np.where(~np.isin(idx_all, idx_remove))[0])
        for key in output.keys():
            output_no_duplicates[key] = [output[key][i] for i in idx_keep]
        print("%d duplicates" % len(idx_remove))
        return output_no_duplicates
    else:
        print("0 duplicates")
        return output


def remove_inaccurate_georef(output, accuracy):
    """
    Function to remove from the output dictionnary entries containing shorelines
    that were mapped on images with inaccurate georeferencing:
        - RMSE > accuracy for Landsat images
        - failed geometric test for Sentinel images (flagged with -1)

    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding metadata
        accuracy: int
            minimum horizontal georeferencing accuracy (metres) for a shoreline to be accepted

    Returns:
    -----------
        output_filtered: dict
            contains the updated dictionary

    """
    # find indices of shorelines to be removed
    idx = []
    for i in range(len(output["geoaccuracy"])):
        geoacc = output["geoaccuracy"][i]
        if geoacc in ["PASSED", "FAILED"]:
            if geoacc == "PASSED":
                idx.append(i)
        else:
            if geoacc <= accuracy:
                idx.append(i)
    output_filtered = dict([])
    for key in output.keys():
        output_filtered[key] = [output[key][i] for i in idx]
    print("%d bad georef" % (len(output["geoaccuracy"]) - len(idx)))
    return output_filtered


def get_nearest_datapoint(dates, dates_ts, values_ts):
    """
    Retrieves the nearest data points from a time-series for a given set of target dates.
    This function handles exact date matches efficiently and finds the closest date when
    an exact match isn't present, using a binary search mechanism for improved performance.

    Ensure that both `dates` and `dates_ts` are datetime objects and are either both timezone-aware
    or both timezone-naive. The function checks if the provided time-series covers the range
    of the target dates and adjusts for edge cases where target dates might align exactly
    with the earliest or latest entries in the time-series.

    Arguments:
    -----------
    dates : list of datetime.datetime
        Target dates for which the nearest data points in the time-series are desired.
    dates_ts : list of datetime.datetime
        Dates in the time-series from which data points will be extracted.
    values_ts : np.array
        Values corresponding to each date in `dates_ts`.

    Returns:
    -----------
    values : np.array
        An array of values from `values_ts` that are closest to each date in `dates`.

    """

    # get closest point to each date (handles exact matches and uses bisect for efficient search)
    indices = [bisect.bisect_left(dates_ts, date) for date in dates]
    temp = []
    for idx, date in zip(indices, dates):
        if idx < len(dates_ts) and dates_ts[idx] == date:
            # Exact match found
            temp.append(values_ts[idx])
        elif idx == 0:
            # Before the first element (shouldn't occur due to range check)
            temp.append(values_ts[0])
        elif idx == len(dates_ts):
            # After the last element (shouldn't occur due to range check)
            temp.append(values_ts[-1])
        else:
            # Find the nearest of the two possible elements
            prev_date = dates_ts[idx - 1]
            next_date = dates_ts[idx]
            if (date - prev_date) <= (next_date - date):
                temp.append(values_ts[idx - 1])
            else:
                temp.append(values_ts[idx])

    return np.array(temp)


def get_closest_datapoint(dates, dates_ts, values_ts):
    """
    Extremely efficient script to get closest data point to a set of dates from a very
    long time-series (e.g., 15-minutes tide data, or hourly wave data)

    Make sure that dates and dates_ts are in the same timezone (also aware or naive)

    KV WRL 2020

    Arguments:
    -----------
    dates: list of datetimes
        dates at which the closest point from the time-series should be extracted
    dates_ts: list of datetimes
        dates of the long time-series
    values_ts: np.array
        array with the values of the long time-series (tides, waves, etc...)

    Returns:
    -----------
    values: np.array
        values corresponding to the input dates

    """

    # check if the time-series cover the dates
    if dates[0] < dates_ts[0] or dates[-1] > dates_ts[-1]:
        raise Exception("Time-series do not cover the range of your input dates")

    # get closest point to each date (no interpolation)
    temp = []

    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start

    for i, date in enumerate(dates):
        print(
            "\rExtracting closest points: %d%%" % int((i + 1) * 100 / len(dates)),
            end="",
        )
        temp.append(
            values_ts[find(min(item for item in dates_ts if item > date), dates_ts)]
        )
    values = np.array(temp)

    return values


###################################################################################################
# GEODATAFRAMES AND READ/WRITE GEOJSON
###################################################################################################
def polygon_from_geojson(fn):
    """
    Extracts coordinates from a .kml file.

    KV WRL 2023

    Arguments:
    -----------
    fn: str
        filepath + filename of the geojson file to be read

    Returns:
    -----------
    polygon: list
        coordinates extracted from the .geojson file

    """

    # read .geojson file
    gdf = gpd.read_file(fn, driver="GeoJSON")
    coords = np.array(gdf.iloc[0]["geometry"].exterior.coords)
    polygon = [[[_[0], _[1]] for _ in coords]]
    return polygon


def polygon_from_kml(fn):
    """
    Extracts coordinates from a .kml file.

    KV WRL 2018

    Arguments:
    -----------
    fn: str
        filepath + filename of the kml file to be read

    Returns:
    -----------
    polygon: list
        coordinates extracted from the .kml file

    """

    # read .kml file
    with open(fn) as kmlFile:
        doc = kmlFile.read()
    # parse to find coordinates field
    str1 = "<coordinates>"
    str2 = "</coordinates>"
    subdoc = doc[doc.find(str1) + len(str1) : doc.find(str2)]
    coordlist = subdoc.split("\n")
    # read coordinates
    polygon = []
    for i in range(1, len(coordlist) - 1):
        polygon.append(
            [float(coordlist[i].split(",")[0]), float(coordlist[i].split(",")[1])]
        )

    return [polygon]


def transects_from_geojson(filename):
    """
    Reads transect coordinates from a .geojson file.

    Arguments:
    -----------
    filename: str
        contains the path and filename of the geojson file to be loaded

    Returns:
    -----------
    transects: dict
        contains the X and Y coordinates of each transect

    """

    gdf = gpd.read_file(filename, driver="GeoJSON")
    transects = dict([])
    for i in gdf.index:
        transects[gdf.loc[i, "name"]] = np.array(gdf.loc[i, "geometry"].coords)

    print("%d transects have been loaded" % len(transects.keys()))
    print("coordinates are in epsg:%d" % gdf.crs.to_epsg())

    return transects


def create_geometry(
    geomtype: str, shoreline: List[List[float]]
) -> Optional[Union[geometry.LineString, geometry.MultiPoint]]:
    """
    Creates geometry based on geomtype and shoreline data.

    Parameters:
    -----------
    geomtype: str
        Type of geometry ('lines' or 'points').
    shoreline: List[List[float]]
        List of shoreline coordinates.
        Ex: [[0,1],[1,0]]

    Returns:
    --------
    Union[geometry.LineString, geometry.MultiPoint, None]
        The created geometry or None if invalid.
    """
    if geomtype == "lines" and len(shoreline) >= 2:
        return geometry.LineString(shoreline)
    elif geomtype == "points" and len(shoreline) > 0:
        return geometry.MultiPoint([(coord[0], coord[1]) for coord in shoreline])
    return None


def create_gdf(
    shoreline: List[List[float]],
    date: datetime,
    satname: str,
    geoaccuracy: Union[float, str],
    cloud_cover: float,
    idx: int,
    geomtype: str,
) -> Optional[gpd.GeoDataFrame]:
    """
    Creates a GeoDataFrame for a given shoreline and its attributes.

    Parameters:
    -----------
    shoreline: List[List[float]]
        List of shoreline coordinates.
    date: datetime
        Date associated with the shoreline.
    satname: str
        Satellite name.
    geoaccuracy: float or string
        Geo accuracy value, can be float for landsat or PASSED/FAILED for S2
    cloud_cover: float
        Cloud cover value.
    idx: int
        Index for the GeoDataFrame.
    geomtype: str
        Type of geometry ('lines' or 'points').

    Returns:
    --------
    Optional[gpd.GeoDataFrame]
        The created GeoDataFrame or None if invalid.
    """
    geom = create_geometry(geomtype, shoreline)
    if geom:
        # Creating a GeoDataFrame directly with all attributes
        data = {
            "date": [date.strftime("%Y-%m-%d %H:%M:%S")],
            "satname": [satname],
            "geoaccuracy": [geoaccuracy],
            "cloud_cover": [cloud_cover],
        }
        gdf = gpd.GeoDataFrame(data, geometry=[geom], index=[idx])
        return gdf
    return None


def output_to_gdf(
    output: Dict[
        str, Union[List[List[List[float]]], List[datetime], List[str], List[float]]
    ],
    geomtype: str,
) -> gpd.GeoDataFrame:
    """
    Converts output data to a GeoDataFrame based on the specified geomtype.

    Parameters:
    -----------
    output : Dict[str, Union[List[List[List[float]]], List[datetime], List[str], List[float]]]
        A dictionary containing:
        - 'shorelines': List of shorelines where each shoreline is a list of coordinates.
        - 'dates': List of datetime objects associated with each shoreline.
        - 'satname': List of satellite names for each shoreline.
        - 'geoaccuracy': List of geo accuracy values for each shoreline.
        - 'cloud_cover': List of cloud cover values for each shoreline.
    geomtype : str
        Type of geometry to be created. Must be either 'lines' or 'points'.

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the consolidated data from the output dictionary.

    Raises:
    -------
    Exception
        If the provided geomtype is not 'lines' or 'points'.
    """
    if geomtype not in ["lines", "points"]:
        raise Exception(
            f"geomtype {geomtype} is not an option, choose between lines or points"
        )

    # create a list of geodataframes and filter out geodataframes that didn't contain shorelines
    # we use filter(None,shoreline_gdf) to not include any empty shorelines which is indicated by create_gdf returning None
    gdf_list = list(
        filter(
            lambda gdf: not gdf.empty if gdf is not None else False,
            map(
                lambda idx: create_gdf(
                    output["shorelines"][idx],
                    output["dates"][idx],
                    output["satname"][idx],
                    output["geoaccuracy"][idx],
                    output["cloud_cover"][idx],
                    idx,
                    geomtype,
                ),
                range(len(output["shorelines"])),
            ),
        )
    )

    return pd.concat(gdf_list, ignore_index=True)


# def output_to_gdf(output, geomtype):
#     """
#     Saves the mapped shorelines as a gpd.GeoDataFrame

#     KV WRL 2018

#     Arguments:
#     -----------
#     output: dict
#         contains the coordinates of the mapped shorelines + attributes
#     geomtype: str
#         'lines' for LineString and 'points' for Multipoint geometry

#     Returns:
#     -----------
#     gdf_all: gpd.GeoDataFrame
#         contains the shorelines + attributes

#     """
#     # loop through the mapped shorelines
#     gdf_list = []
#     for i in range(len(output["shorelines"])):
#         # skip if there shoreline is empty
#         if len(output["shorelines"][i]) == 0:
#             continue
#         else:
#             # save the geometry depending on the linestyle
#             if geomtype == "lines":
#                 # linestrings must consist of 2 or more points
#                 if len(output["shorelines"][i]) < 2:
#                     continue
#                 geom = geometry.LineString(output["shorelines"][i])
#             elif geomtype == "points":
#                 coords = output["shorelines"][i]
#                 geom = geometry.MultiPoint(
#                     [(coords[_, 0], coords[_, 1]) for _ in range(coords.shape[0])]
#                 )
#             else:
#                 raise Exception(
#                     "geomtype %s is not an option, choose between lines or points"
#                     % geomtype
#                 )
#             # save into geodataframe with attributes
#             gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
#             gdf.index = [i]
#             gdf.loc[i, "date"] = output["dates"][i].strftime("%Y-%m-%d %H:%M:%S")
#             gdf.loc[i, "satname"] = output["satname"][i]
#             gdf.loc[i, "geoaccuracy"] = output["geoaccuracy"][i]
#             gdf.loc[i, "cloud_cover"] = output["cloud_cover"][i]
#             gdf_list.append(gdf)

#     # concatenate all GeoDataFrames in the list into a single GeoDataFrame
#     gdf_all = pd.concat(gdf_list, ignore_index=True)

#     return gdf_all

# # loop through the mapped shorelines
# counter = 0
# gdf_all = None
# for i in range(len(output['shorelines'])):
#     # skip if there shoreline is empty
#     if len(output['shorelines'][i]) == 0:
#         continue
#     else:
#         # save the geometry depending on the linestyle
#         if geomtype == 'lines':
#             # linestrings must consist of 2 or more points
#             if len(output['shorelines'][i]) < 2:
#                 continue
#             geom = geometry.LineString(output['shorelines'][i])
#         elif geomtype == 'points':
#             coords = output['shorelines'][i]
#             geom = geometry.MultiPoint([(coords[_,0], coords[_,1]) for _ in range(coords.shape[0])])
#         else:
#             raise Exception('geomtype %s is not an option, choose between lines or points'%geomtype)
#         # save into geodataframe with attributes
#         gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
#         gdf.index = [i]
#         gdf.loc[i,'date'] = output['dates'][i].strftime('%Y-%m-%d %H:%M:%S')
#         gdf.loc[i,'satname'] = output['satname'][i]
#         gdf.loc[i,'geoaccuracy'] = output['geoaccuracy'][i]
#         gdf.loc[i,'cloud_cover'] = output['cloud_cover'][i]
#         # store into geodataframe
#         if counter == 0:
#             gdf_all = gdf
#         else:
#             gdf_all = gdf_all.append(gdf)
#         counter = counter + 1

# return gdf_all


def transects_to_gdf(transects):
    """
    Saves the shore-normal transects as a gpd.GeoDataFrame

    KV WRL 2018

    Arguments:
    -----------
    transects: dict
        contains the coordinates of the transects

    Returns:
    -----------
    gdf_all: gpd.GeoDataFrame


    """
    # loop through the mapped shorelines
    gdf_list = []
    for i, key in enumerate(list(transects.keys())):
        # save the geometry + attributes
        geom = geometry.LineString(transects[key])
        gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
        gdf.index = [i]
        gdf.loc[i, "name"] = key
        gdf_list.append(gdf)

    # concatenate all GeoDataFrames in the list into a single GeoDataFrame
    gdf_all = pd.concat(gdf_list, ignore_index=True)

    return gdf_all

    # # loop through the mapped shorelines
    # for i,key in enumerate(list(transects.keys())):
    #     # save the geometry + attributes
    #     geom = geometry.LineString(transects[key])
    #     gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(geom))
    #     gdf.index = [i]
    #     gdf.loc[i,'name'] = key
    #     # store into geodataframe
    #     if i == 0:
    #         gdf_all = gdf
    #     else:
    #         gdf_all = gdf_all.append(gdf)

    # return gdf_all


def smallest_rectangle(polygon):
    """
    Converts a polygon to the smallest rectangle polygon with sides parallel
    to coordinate axes.

    KV WRL 2020

    Arguments:
    -----------
    polygon: list of coordinates
        pair of coordinates for 5 vertices, in clockwise order,
        first and last points must match

    Returns:
    -----------
    polygon: list of coordinates
        smallest rectangle polygon

    """

    multipoints = geometry.Polygon(polygon[0])
    polygon_geom = multipoints.envelope
    coords_polygon = np.array(polygon_geom.exterior.coords)
    polygon_rect = [[[_[0], _[1]] for _ in coords_polygon]]
    return polygon_rect


def make_animation_mp4(filepath_images, fps, fn_out):
    "function to create an animation with the saved figures"
    import imageio

    with imageio.get_writer(fn_out, mode="I", fps=fps) as writer:
        filenames = os.listdir(filepath_images)
        # order chronologically
        filenames = np.sort(filenames)
        for i in range(len(filenames)):
            image = imageio.imread(os.path.join(filepath_images, filenames[i]))
            writer.append_data(image)
    print(
        "Animation has been generated (using %d frames per second) and saved at %s"
        % (fps, fn_out)
    )


def compare_timeseries(ts, gt, key, settings):
    if key not in gt.keys():
        raise Exception("transect name %s does not exist in grountruth file" % key)
    # remove nans
    chainage = np.array(ts[key])
    idx_nan = np.isnan(chainage)
    dates_nonans = [ts["dates"][k].to_pydatetime() for k in np.where(~idx_nan)[0]]
    satnames_nonans = [ts["satname"][k] for k in np.where(~idx_nan)[0]]
    chain_nonans = chainage[~idx_nan]
    # define satellite and survey time-series
    chain_sat_dm = chain_nonans
    chain_sur_dm = gt[key]["chainages"]
    # plot the time-series
    fig = plt.figure(figsize=[15, 8], tight_layout=True)
    gs = gridspec.GridSpec(2, 3)
    ax0 = fig.add_subplot(gs[0, :])
    ax0.grid(which="major", linestyle=":", color="0.5")
    ax0.plot(gt[key]["dates"], chain_sur_dm, "-o", mfc="w", ms=3, label="in situ")
    ax0.plot(dates_nonans, chain_sat_dm, "-o", mfc="w", ms=3, label="satellite")
    ax0.set(
        title="Transect " + key,
        xlim=[
            dates_nonans[0] - timedelta(days=30),
            dates_nonans[-1] + timedelta(days=30),
        ],
    )  # ,ylim=sett['lims'])
    ax0.legend(loc="upper left")

    # interpolate surveyed data around satellite data based on the parameters (min_days and max_days)
    chain_int = np.nan * np.ones(len(dates_nonans))
    for k, date in enumerate(dates_nonans):
        # compute the days distance for each satellite date
        days_diff = np.array([(_ - date).days for _ in gt[key]["dates"]])
        # if nothing within max_days put a nan
        if np.min(np.abs(days_diff)) > settings["max_days"]:
            chain_int[k] = np.nan
        else:
            # if a point within min_days, take that point (no interpolation)
            if np.min(np.abs(days_diff)) < settings["min_days"]:
                idx_closest = np.where(np.abs(days_diff) == np.min(np.abs(days_diff)))
                chain_int[k] = float(gt[key]["chainages"][idx_closest[0][0]])
            else:  # otherwise, between min_days and max_days, interpolate between the 2 closest points
                if sum(days_diff > 0) == 0:
                    break
                idx_after = np.where(days_diff > 0)[0][0]
                idx_before = idx_after - 1
                x = [
                    gt[key]["dates"][idx_before].toordinal(),
                    gt[key]["dates"][idx_after].toordinal(),
                ]
                y = [gt[key]["chainages"][idx_before], gt[key]["chainages"][idx_after]]
                f = interpolate.interp1d(x, y, bounds_error=True)
                try:
                    chain_int[k] = float(f(date.toordinal()))
                except:
                    chain_int[k] = np.nan
    # remove nans again
    idx_nan = np.isnan(chain_int)
    chain_sat = chain_nonans[~idx_nan]
    chain_sur = chain_int[~idx_nan]
    dates_sat = [dates_nonans[k] for k in np.where(~idx_nan)[0]]
    satnames = [satnames_nonans[k] for k in np.where(~idx_nan)[0]]
    if len(chain_sat) < 8 or len(chain_sur) < 8:
        return chain_sat, chain_sur, satnames, fig
    # error statistics
    slope, intercept, rvalue, pvalue, std_err = stats.linregress(chain_sur, chain_sat)
    R2 = rvalue**2
    ax0.text(
        0,
        1,
        "R2 = %.2f" % R2,
        bbox=dict(boxstyle="square", facecolor="w", alpha=1),
        transform=ax0.transAxes,
    )
    chain_error = chain_sat - chain_sur
    rmse = np.sqrt(np.mean((chain_error) ** 2))
    mean = np.mean(chain_error)
    std = np.std(chain_error)
    q90 = np.percentile(np.abs(chain_error), 90)

    # 1:1 plot
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.axis("equal")
    ax1.grid(which="major", linestyle=":", color="0.5")
    for k, sat in enumerate(list(np.unique(satnames))):
        idx = np.where([_ == sat for _ in satnames])[0]
        ax1.plot(
            chain_sur[idx],
            chain_sat[idx],
            "o",
            ms=4,
            mfc="C" + str(k),
            mec="C" + str(k),
            alpha=0.7,
            label=sat,
        )
    ax1.legend(loc=4)
    ax1.plot(
        [ax1.get_xlim()[0], ax1.get_ylim()[1]],
        [ax1.get_xlim()[0], ax1.get_ylim()[1]],
        "k--",
        lw=2,
    )
    ax1.set(xlabel="survey [m]", ylabel="satellite [m]")

    # boxplots
    ax2 = fig.add_subplot(gs[1, 1])
    data = []
    median_data = []
    n_data = []
    ax2.yaxis.grid()
    for k, sat in enumerate(list(np.unique(satnames))):
        idx = np.where([_ == sat for _ in satnames])[0]
        data.append(chain_error[idx])
        median_data.append(np.median(chain_error[idx]))
        n_data.append(len(chain_error[idx]))
    bp = ax2.boxplot(data, 0, "k.", labels=list(np.unique(satnames)), patch_artist=True)
    for median in bp["medians"]:
        median.set(color="k", linewidth=1.5)
    for j, boxes in enumerate(bp["boxes"]):
        boxes.set(facecolor="C" + str(j))
        ax2.text(
            j + 1,
            median_data[j] + 1,
            "%.1f" % median_data[j],
            horizontalalignment="center",
            fontsize=12,
        )
        ax2.text(
            j + 1 + 0.35,
            median_data[j] + 1,
            ("n=%.d" % int(n_data[j])),
            ha="center",
            va="center",
            fontsize=12,
            rotation="vertical",
        )
    ax2.set(ylabel="error [m]", ylim=settings["lims"])

    # histogram
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.grid(which="major", linestyle=":", color="0.5")
    ax3.axvline(x=0, ls="--", lw=1.5, color="k")
    binwidth = settings["binwidth"]
    bins = np.arange(min(chain_error), max(chain_error) + binwidth, binwidth)
    density = plt.hist(
        chain_error, bins=bins, density=True, color="0.6", edgecolor="k", alpha=0.5
    )
    mu, std = stats.norm.fit(chain_error)
    pval = stats.normaltest(chain_error)[1]
    xlims = ax3.get_xlim()
    x = np.linspace(xlims[0], xlims[1], 100)
    p = stats.norm.pdf(x, mu, std)
    ax3.plot(x, p, "r-", linewidth=1)
    ax3.set(xlabel="error [m]", ylabel="pdf", xlim=settings["lims"])
    str_stats = " rmse = %.1f\n mean = %.1f\n std = %.1f\n q90 = %.1f" % (
        rmse,
        mean,
        std,
        q90,
    )
    ax3.text(0, 0.98, str_stats, va="top", transform=ax3.transAxes)

    return chain_sat, chain_sur, satnames, fig


def ordinal(n: int):
    """
    Produces ordinal number suffix (1st 2nd 3rd) for readability when downloading imagery.

    FM UofGlasgow 2024

    Arguments:
    -----------
    n: int
        number usually relating to an image in a list

    Returns:
    -----------
    ordnum: string
        original number of image but as a string with correct ordinal ending

    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    ordnum = str(n) + suffix
    return ordnum
