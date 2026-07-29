# pylint: disable=line-too-long
"""
This module contains all the functions needed for extracting satellite-derived
shorelines (SDS)

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# standard library imports
import datetime
import os
import pdb
import logging
import traceback
from typing import Tuple, List, Union, Dict, Any

# related third party imports
import geopandas as gpd
import joblib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib import gridspec
from pylab import ginput
from shapely.geometry import LineString
import skimage.draw as skdraw
from skimage import filters, measure, morphology
from shapely.geometry import LineString, Point
from skimage.filters import threshold_multiotsu
from tqdm.auto import tqdm

# local application/library specific imports
from coastsat import SDS_preprocess, SDS_sar_model, SDS_tools
from coastsat.SDS_tools import create_geometry
from coastsat.SDS_download import release_logger, setup_logger
from coastsat.classification import models, training_data, training_sites

# set numpy error handling
np.seterr(all="ignore")  # raise/ignore divisions by 0 and nans


def filter_contours(
    contours: List[np.ndarray],
    georef: dict,
    output_epsg: int,
    image_epsg: int,
    min_length_sl: float,
) -> np.ndarray:
    """
    Filters a list of contour lines by removing those shorter than a specified minimum length,
    and converts coordinates from pixel space to a desired spatial reference system (EPSG).

    Args:
        contours (List[np.ndarray]): List of contour arrays with pixel coordinates.
        georef (dict): Georeferencing information for pixel-to-world coordinate transformation.
        output_epsg (int): EPSG code of the desired output coordinate system.
        image_epsg (int): EPSG code of the image's current coordinate system.
        min_length_sl (float): Minimum length (in output EPSG units) for a contour to be kept.

    Returns:
        np.ndarray: A 2D array of concatenated x and y coordinates of all valid contours
                    in the output EPSG coordinate system.
    """
    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(contours_world, image_epsg, output_epsg)

    # 1. Remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)  # shapely LineString structure
        if a.length >= min_length_sl:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points, contours_long[k][:, 0])
        y_points = np.append(y_points, contours_long[k][:, 1])
    contours_array = np.transpose(np.array([x_points, y_points]))

    shoreline = contours_array  # shoreline is spatially referenced in the output_epsg

    return shoreline


def should_skip_image(
    cloud_cover_combined: float,
    cloud_cover: float,
    settings: Dict[str, float],
    logger: logging.Logger,
    satname: str,
    date_str: str,
) -> bool:
    """
    Determine whether to skip a satellite image based on cloud metrics.

    Parameters:
    ----------
    cloud_cover_combined : float
        Proportion of cloud + no-data pixels over the total image area.
    cloud_cover : float
        Proportion of cloud pixels over valid (non-no-data) pixels.
    settings : Dict[str, float]
        Configuration dictionary with at least the key:
            - "cloud_thresh": float, threshold above which the image is skipped.
    logger : logging.Logger
        Logger object to record warnings.
    satname : str
        Name of the satellite (e.g., "S1", "S2").
    date_str : str
        Date string representing the acquisition date of the image.

    Returns:
    -------
    bool
        True if the image should be skipped due to excessive cloud or no-data pixels, False otherwise.
    """
    if cloud_cover_combined > 0.99:
        logger.warning(
            f"{satname} {date_str}: Skipped due to >99% cloud + no-data pixels"
        )
        return True
    if cloud_cover > settings["cloud_thresh"]:
        logger.warning(
            f"{satname} {date_str}: Skipped due to cloud cover {cloud_cover:.2%} "
            f"> threshold {settings['cloud_thresh']:.2%}"
        )
        return True
    return False


# Polarization the SAR shoreline detection thresholds when the scene carries only one
# usable band. CoastSat has always thresholded VH, so keeping it as the default leaves
# existing results unchanged now that several polarizations can be stacked into im_ms.
# A dual-polarization scene thresholds the mean of VV and VH instead, unless
# settings['sar_detection_polarization'] names a band explicitly.
SAR_DETECTION_POLARIZATION = "VH"


def get_sar_detection_polarization(settings=None):
    """
    Returns the polarization settings ask the Otsu path to threshold, or None.

    None means "no explicit request", which leaves the default behaviour of
    get_sar_otsu_image in place rather than forcing a band.

    Arguments:
    -----------
    settings: dict, optional
        may contain 'sar_detection_polarization', e.g. 'VV'

    Returns:
    -----------
    str or None: the requested polarization, upper-cased, or None if unset

    """
    if not settings:
        return None
    polarization = settings.get("sar_detection_polarization")
    if not polarization:
        return None
    return str(polarization).strip().upper()


def warn_unknown_sar_detection_polarization(settings, logger=None):
    """
    Warns once per run when settings ask for a polarization CoastSat cannot download.

    A typo would otherwise be indistinguishable from the default, because a request this
    code cannot honour falls back to the usual band (see get_sar_otsu_image).

    Arguments:
    -----------
    settings: dict
        may contain 'sar_detection_polarization'
    logger: logging.Logger, optional

    Returns:
    -----------
    str or None: the requested polarization, as get_sar_detection_polarization returns it

    """
    requested = get_sar_detection_polarization(settings)
    if (
        requested is not None
        and requested not in SDS_tools.SAR_POLARIZATIONS
        and logger is not None
    ):
        logger.warning(
            f"S1: settings['sar_detection_polarization'] is '{requested}', which is not "
            f"one of {list(SDS_tools.SAR_POLARIZATIONS)}; the Otsu fallback will "
            "threshold the usual band instead"
        )
    return requested


def get_sar_channel_bands(fn):
    """
    Returns what each channel of im_ms holds, as preprocess_image built it.

    For a dual-polarization scene preprocess_image replaces the raw polarization stack
    with the [VV, VH, VV-VH] composite, so the channels are the composite bands and no
    longer line up one-to-one with `fn`. Otherwise the channels are the polarizations of
    `fn`, in that order.

    Arguments:
    -----------
    fn: str or list of str
        the image file(s) of the scene, as returned by SDS_tools.get_filenames

    Returns:
    -----------
    list: the band held by each channel of im_ms

    """
    if isinstance(fn, str):
        fn = [fn]
    polarizations = [SDS_tools.get_polarization_from_filename(f) for f in fn]

    if "VV" in polarizations and "VH" in polarizations:
        return list(SDS_tools.SAR_COMPOSITE_BANDS)
    return polarizations


def get_sar_band_index(fn, polarization: str = SAR_DETECTION_POLARIZATION) -> int:
    """
    Returns the channel of im_ms holding the polarization used for SAR detection.

    `fn` is the list of files read_sar_image() stacked, so its order is the channel
    order. Falls back to the first channel when the polarization is not present,
    which covers scenes downloaded before multi-polarization support.

    Arguments:
    -----------
    fn: str or list of str
        the image file(s) of the scene, as returned by SDS_tools.get_filenames
    polarization: str
        polarization to threshold, defaults to VH

    Returns:
    -----------
    int: index of the channel to threshold

    """
    if isinstance(fn, str):
        fn = [fn]
    for index, file_path in enumerate(fn):
        if SDS_tools.get_polarization_from_filename(file_path) == polarization:
            return index
    return 0


def get_sar_otsu_image(im_ms, fn, settings=None):
    """
    Returns the 2D image the SAR Otsu threshold runs on.

    For a dual-polarization scene im_ms is the [VV, VH, VV-VH] composite and the
    threshold runs on the mean of VV and VH (in dB). Note VV-VH is deliberately left
    out of that mean: VV + VH + (VV - VH) = 2*VV, so including it would cancel VH
    and reduce the result to a rescaling of VV alone.

    Scenes with a single polarization, i.e. those downloaded before dual-polarization
    support, keep thresholding that one band so their results are unchanged.

    settings['sar_detection_polarization'] overrides both: it thresholds that band
    directly whenever the scene carries it. A scene that does not carry it falls back to
    the behaviour above rather than failing, so one odd scene in a run still maps.

    Arguments:
    -----------
    im_ms: np.ndarray
        (H, W, n) composite or polarization stack, as returned by preprocess_image
    fn: str or list of str
        the image file(s) of the scene, used to identify the polarizations
    settings: dict, optional
        may contain 'sar_detection_polarization' to threshold one band explicitly

    Returns:
    -----------
    np.ndarray: the (H, W) image to threshold

    """
    bands = get_sar_channel_bands(fn)

    requested = get_sar_detection_polarization(settings)
    if requested is not None and requested in bands:
        return im_ms[:, :, bands.index(requested)]

    if "VV" in bands and "VH" in bands:
        return (im_ms[:, :, bands.index("VV")] + im_ms[:, :, bands.index("VH")]) / 2.0

    return im_ms[:, :, get_sar_band_index(fn)]


def describe_sar_otsu_image(fn, settings=None) -> str:
    """
    Describes what get_sar_otsu_image() returns for `fn`, for logging.

    Reports the band actually thresholded, so a 'sar_detection_polarization' the scene
    does not carry is visible in the log as the band that replaced it.
    """
    bands = get_sar_channel_bands(fn)

    requested = get_sar_detection_polarization(settings)
    if requested is not None and requested in bands:
        return requested

    if "VV" in bands and "VH" in bands:
        return "the mean of VV and VH"

    return str(bands[get_sar_band_index(fn)])


# How SAR water is segmented: "model" runs the ONNX land/water model on the
# [VV, VH, VV-VH] composite, "otsu" forces the legacy global threshold. The model is the
# default; scenes it cannot handle fall back to Otsu on their own (see load_sar_segmenter).
SAR_SEGMENTATION_DEFAULT = "model"


def load_sar_segmenter(settings, logger=None):
    """
    Loads the SAR segmentation model once for a whole run, or None to use Otsu.

    Returns None rather than raising whenever the model cannot be used - onnxruntime not
    installed, the .onnx missing, its download failing offline, settings asking for
    Otsu - so that a run always produces shorelines, using the legacy threshold where
    the model is unavailable.

    Arguments:
    -----------
    settings: dict
        may contain 'sar_segmentation' ("model" or "otsu") and 'sar_model_path'
    logger: logging.Logger, optional
        records which segmentation method the run will use

    Returns:
    -----------
    tuple or None: (onnxruntime session, model spec), or None to threshold with Otsu

    """
    method = str(settings.get("sar_segmentation", SAR_SEGMENTATION_DEFAULT)).lower()

    if method == "otsu":
        if logger is not None:
            logger.info(
                "S1: settings['sar_segmentation'] is 'otsu', thresholding instead of "
                "running the segmentation model"
            )
        return None

    override = settings.get("sar_model_path")
    model_path = SDS_sar_model.get_sar_model_path(settings)
    try:
        # None means the default model, which load_sar_model may download on first
        # use; an explicit override path is never downloaded
        session, spec = SDS_sar_model.load_sar_model(model_path if override else None)
    except SDS_sar_model.SarModelUnavailable as e:
        if logger is not None:
            logger.warning(
                f"S1: falling back to Otsu thresholding for every scene, because {e}"
            )
        return None

    if logger is not None:
        logger.info(
            f"S1: segmenting land/water with {os.path.basename(model_path)} "
            f"(channels {list(spec['channel_order'])})"
        )
    return session, spec


def segment_sar_water(im_ms, fn, segmenter, settings, logger=None, date_str=""):
    """
    Segments water in one SAR scene with the model, or returns None to use Otsu.

    Arguments:
    -----------
    im_ms: np.ndarray
        (H, W, 3) [VV, VH, VV-VH] composite in dB, as returned by preprocess_image
    fn: list of str
        the polarization files of the scene, used to read the validity mask
    segmenter: tuple or None
        (session, spec) as returned by load_sar_segmenter
    settings: dict
        may contain 'sar_water_threshold'
    logger: logging.Logger, optional
    date_str: str
        image date, for the log message

    Returns:
    -----------
    tuple or None:
        np.ndarray: (H, W) boolean water mask
        np.ndarray: (H, W) boolean validity mask
        np.ndarray: (H, W) float32 P(water), the probability the mask was cut from
        or None if the model could not segment this scene

    """
    if segmenter is None:
        return None

    session, spec = segmenter
    try:
        valid = SDS_tools.read_sar_valid_mask(fn)
        im_water, im_prob = SDS_sar_model.segment_water(
            im_ms,
            valid,
            session,
            spec,
            settings.get("sar_water_threshold", SDS_sar_model.WATER_THRESHOLD),
        )
    except SDS_sar_model.SarModelUnavailable as e:
        if logger is not None:
            logger.warning(
                f"S1 {date_str}: falling back to Otsu thresholding, because {e}"
            )
        return None

    return im_water, valid, im_prob


def save_sar_probability(im_prob, valid, fn, logger=None, date_str=""):
    """
    Writes P(water) of a model-segmented scene as a single-band float32 GeoTIFF.

    The raster goes to S1/prob/, a sibling of the polarization folders, named like
    the scene files with 'prob' in place of the polarization token (the _dupN
    suffix, when present, is kept so the file pairs up with its scene). It copies
    the grid of the first polarization file, so it overlays the inputs exactly.
    Invalid pixels are written as NaN, which is also the declared nodata value -
    unlike the polarization rasters, where nodata is 0.0, a value P(water) can
    genuinely take.

    A failure to write is logged and swallowed: the probability raster is a
    by-product, and losing it must not cost the scene its shoreline.

    Arguments:
    -----------
    im_prob: np.ndarray
        (H, W) P(water), as returned by segment_sar_water
    valid: np.ndarray
        (H, W) boolean validity mask; invalid pixels are written as NaN
    fn: str or list of str
        the polarization files of the scene, used for the grid and the name
    logger: logging.Logger, optional
    date_str: str
        image date, for the log message

    Returns:
    -----------
    str or None: path of the GeoTIFF written, or None if it could not be written

    """
    from osgeo import gdal

    if isinstance(fn, str):
        fn = [fn]

    try:
        source = gdal.Open(fn[0], gdal.GA_ReadOnly)
        if source is None:
            raise IOError(f"could not open {fn[0]} to copy its georeferencing")

        prob_dir = os.path.join(os.path.dirname(os.path.dirname(fn[0])), "prob")
        os.makedirs(prob_dir, exist_ok=True)
        filename = SDS_tools.swap_polarization_in_filename(
            os.path.basename(fn[0]), "prob"
        )
        prob_path = os.path.join(prob_dir, filename)

        data = np.asarray(im_prob, dtype=np.float32).copy()
        data[~valid] = np.nan

        driver = gdal.GetDriverByName("GTiff")
        height, width = data.shape
        dataset = driver.Create(
            prob_path,
            width,
            height,
            1,
            gdal.GDT_Float32,
            options=["COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"],
        )
        dataset.SetGeoTransform(source.GetGeoTransform())
        dataset.SetProjection(source.GetProjection())
        band = dataset.GetRasterBand(1)
        band.SetNoDataValue(float("nan"))
        band.WriteArray(data)
        dataset.FlushCache()
        dataset = None
        source = None
    except Exception as e:
        if logger is not None:
            logger.warning(
                f"S1 {date_str}: could not save the P(water) GeoTIFF, because {e}"
            )
        return None

    return prob_path


def compute_sar_water_fraction(im_water, region):
    """
    Fraction of `region` pixels classified as water, or NaN when the region is empty.

    Computed over the reference shoreline buffer (intersected with the valid pixels
    on the model path): that is where the shoreline is contoured, so a fraction near
    0 or 1 there means the segmentation found (almost) no land/water interface and
    the mapped shoreline is suspect. Recorded per scene in output['water_fraction'],
    which is what the SAR transect QC filters on (see SDS_transects.sar_qc_keep).

    Arguments:
    -----------
    im_water: np.ndarray
        (H, W) boolean water mask, True for water
    region: np.ndarray
        (H, W) boolean mask of the pixels to compute the fraction over

    Returns:
    -----------
    float: the water fraction in [0, 1], or NaN

    """
    region = np.asarray(region, dtype=bool)
    if not np.any(region):
        return np.nan
    return float(np.mean(np.asarray(im_water, dtype=bool)[region]))


def otsu_threshold(im, min_beach_area_pixels):
    """
    Splits a SAR band into land and water with Otsu, dropping the small objects.

    Arguments:
    -----------
    im: np.ndarray
        (H, W) image to threshold, in dB
    min_beach_area_pixels: int
        smallest object to keep, in *pixels* - settings['min_beach_area'] is in m^2, so
        divide it by pixel_size**2 first (see extract_shorelines)

    Returns:
    -----------
    tuple:
        np.ndarray: (H, W) boolean mask, True above the threshold
        np.ndarray: the Otsu threshold(s), in dB

    """
    # Apply Otsu's thresholding
    threshold = threshold_multiotsu(im, classes=2)  # split into water and land
    otsu = im > min(
        threshold
    )  # only a single threshold but lets pretend there could be more
    otsu_med = morphology.remove_small_objects(
        otsu, min_size=int(min_beach_area_pixels), connectivity=2
    )

    return otsu_med, threshold


def find_shoreline_SAR(otsu_image, ref_shoreline_buffer, georef, image_epsg, settings):
    otsu_med_masked = otsu_image.copy().astype(float)
    otsu_med_masked[~ref_shoreline_buffer] = np.nan
    contours = measure.find_contours(
        otsu_med_masked, 0.5
    )  # returns lists of np.arrays that are row col formatted example [[[y1,x1],[y2,x2],...], [[[y1,x1],[y2,x2],...]]]
    processed_contours = process_contours(contours)

    shoreline = filter_contours(
        processed_contours,
        georef,
        settings["output_epsg"],
        image_epsg,
        settings["min_length_sl"],
    )
    return shoreline


def find_shoreline_optical(
    im_ms,
    im_labels,
    im_ref_buffer,
    cloud_mask,
    cloud_mask_adv,
    im_nodata,
    georef,
    image_epsg,
    satname,
    shoreline_date,
    settings,
    logger=None,
):
    """
    Finds the shoreline using either MNDWI contours or label-based classification,
    based on the number of sand pixels in the reference buffer.

    Args:
        im_ms (np.ndarray): Multispectral image.
        im_labels (np.ndarray): Labeled classification image.
        im_ref_buffer (np.ndarray): Reference buffer mask for shoreline detection.
        cloud_mask (np.ndarray): Cloud mask (bool array).
        cloud_mask_adv (np.ndarray): Advanced cloud mask for final shoreline filtering.
        im_nodata (np.ndarray): Nodata mask.
        georef: Georeferencing metadata.
        image_epsg (int): EPSG code for image projection.
        satname (str): Satellite name for logging.
        shoreline_date (str): Date of shoreline image (for logging).
        settings (dict): Configuration settings.
        logger (logging.Logger, optional): Logger instance for logging messages.

    Returns:
        shoreline (list or array): Extracted shoreline geometry.
        t_mndwi (float): Threshold value used for MNDWI if applicable.
    """
    if logger is None:
        import logging

        logger = logging.getLogger(__name__)

    # Check sand pixels in reference buffer
    if sum(im_labels[im_ref_buffer, 0]) < 50:
        im_mndwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
        logger.info(
            f"{satname} {shoreline_date}: Less than 50 sand pixels detected within reference shoreline buffer. Using find_wl_contours1"
        )
        contours_mwi, t_mndwi = find_wl_contours1(im_mndwi, cloud_mask, im_ref_buffer)
    else:
        logger.info(
            f"{satname} {shoreline_date}: Greater than 50 sand pixels detected within reference shoreline buffer. Using find_wl_contours2"
        )
        contours_mwi, t_mndwi = find_wl_contours2(
            im_ms, im_labels, cloud_mask, im_ref_buffer
        )

    # Process contours into shoreline
    shoreline = process_shoreline(
        contours_mwi,
        cloud_mask_adv,
        im_nodata,
        georef,
        image_epsg,
        settings,
        logger=logger,
    )

    return shoreline, t_mndwi


def compute_cloud_metrics(
    cloud_mask: np.ndarray, im_nodata: np.ndarray
) -> Tuple[float, float, np.ndarray]:
    """
    Compute cloud cover metrics from the given cloud and no-data masks.

    Parameters:
    ----------
    cloud_mask : np.ndarray
        Boolean array indicating initial cloud detection (True = cloud).
    im_nodata : np.ndarray
        Boolean array indicating no-data pixels (True = no data).

    Returns:
    -------
    Tuple[float, float, np.ndarray]
        - cloud_cover_combined: Fraction of cloud pixels over the total image (including no-data).
        - cloud_cover: Fraction of cloud pixels over valid (non-no-data) pixels.
        - cloud_mask_adv: Advanced cloud mask where no-data pixels are excluded.
    """
    cloud_mask = np.asarray(cloud_mask, dtype=bool)
    im_nodata = np.asarray(im_nodata, dtype=bool)

    cloud_cover_combined = np.sum(cloud_mask) / cloud_mask.size
    # remove no data pixels from the cloud mask
    # (for example L7 bands of no data should not be accounted for)
    try:
        cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    except ValueError as exc:
        raise ValueError(
            f"Cloud/no-data mask shape mismatch: cloud_mask {cloud_mask.shape}, "
            f"im_nodata {im_nodata.shape}"
        ) from exc
    valid_pixels = np.sum(~im_nodata)

    # compute updated cloud cover percentage (without no data pixels)
    # Avoid division by zero if the entire image is no-data
    cloud_cover = (
        np.sum(cloud_mask_adv.astype(int)) / valid_pixels.astype(int)
        if valid_pixels > 0
        else 0.0
    )

    return cloud_cover_combined, cloud_cover, cloud_mask_adv


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
        im_ms, georef, cloud_mask, im_extra, im_QA, im_nodata = (
            SDS_preprocess.preprocess_single(
                fn,
                satname,
                settings["cloud_mask_issue"],
                settings["pan_off"],
                collection,
                apply_cloud_mask,
                settings.get("s2cloudless_prob", 60),
            )
        )
        return im_ms, georef, cloud_mask, im_nodata


def load_classifier(
    satname: str, sand_color: str, model_path: str, sklearn_version: str
) -> object:
    """
    Load the appropriate machine learning classifier based on satellite name and sand color.

    Parameters:
    ----------
    satname : str
        Name of the satellite (e.g., "L5", "L7", "L8", "L9", "S1", "S2").
    sand_color : str
        Type of sand reflectance ("dark", "bright", "default", "latest").
    model_path : str
        Path to the directory containing model .pkl files.
    sklearn_version : str
        Version string of scikit-learn (used to pick model compatibility).

    Returns:
    -------
    object
        Loaded classifier object.
    """
    str_new = "_new" if not sklearn_version.startswith("0.20") else ""

    if satname == "S2":
        model_file = f"NN_4classes_S2{str_new}.pkl"
    else:
        if sand_color == "dark":
            model_file = f"NN_4classes_Landsat_dark{str_new}.pkl"
        elif sand_color == "bright":
            model_file = f"NN_4classes_Landsat_bright{str_new}.pkl"
        elif sand_color == "latest":
            model_file = f"NN_4classes_Landsat_latest{str_new}.pkl"
        else:  # default or unrecognized value
            model_file = f"NN_4classes_Landsat{str_new}.pkl"

    full_model_path = os.path.join(model_path, model_file)
    return joblib.load(full_model_path)


def LineString_to_arr(line):
    return np.array(line.coords)


def arr_to_LineString(arr):
    return LineString(arr)


def get_model_locations():
    try:
        import importlib.resources

        filepath_models = os.path.abspath(importlib.resources.files(models))
    except AttributeError:
        from importlib_resources import files

        filepath_models = os.path.abspath(files(models))

    return filepath_models


def load_model(satname: str, settings: dict):
    """
    Loads the appropriate model based on the satellite name and settings.

    Parameters:
    - satname (str): The satellite name (e.g., "L5", "L7", "L8", "L9", "S2").
    - settings (dict): A dictionary containing settings, particularly the "sand_color".

    Returns:
    - clf: The loaded classifier model.
    - pixel_size (int): The pixel size associated with the satellite.
    """
    # get the locations to load the models from
    filepath_models = get_model_locations()

    str_new = "_new" if not sklearn.__version__[:4] == "0.20" else ""
    clf = None

    if satname in ["L5", "L7", "L8", "L9"]:
        pixel_size = 15
        model_filename = None

        if settings["sand_color"] == "dark":
            model_filename = f"NN_4classes_Landsat_dark{str_new}.pkl"
        elif settings["sand_color"] == "bright":
            model_filename = f"NN_4classes_Landsat_bright{str_new}.pkl"
        elif settings["sand_color"] == "default":
            model_filename = f"NN_4classes_Landsat{str_new}.pkl"
        elif settings["sand_color"] == "latest":
            model_filename = f"NN_4classes_Landsat_latest{str_new}.pkl"

        if model_filename:
            clf = joblib.load(os.path.join(filepath_models, model_filename))

    elif satname == "S2":
        pixel_size = 10
        clf = joblib.load(os.path.join(filepath_models, f"NN_4classes_S2{str_new}.pkl"))

    return clf, pixel_size


def convert_gdf_to_array(gdf: gpd.GeoDataFrame) -> list:
    """
    Convert a GeoDataFrame to a list of NumPy arrays.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame to be converted.

    Returns:
        list: The converted list of NumPy arrays.

    Note:
        This function will not work for multi-geometries like multi-polygons.
    """
    new_array = []
    # for each geometry in the gdf, convert it to an array
    for idx in range(len(gdf)):
        array = np.array(gdf.iloc[idx].geometry.exterior.coords)
        new_array.append(array)
    return new_array


def convert_shoreline_to_array(gdf) -> np.ndarray:
    """
    Convert filtered shoreline GeoDataFrame to a numpy array.

    Parameters:
    gdf (GeoDataFrame): The filtered shoreline GeoDataFrame.

    Returns:
    np.ndarray: The numpy array representation of the shoreline coordinates.
    """
    if gdf.empty:
        return np.array([])
    else:
        x_array, y_array = gdf.geometry.iloc[0].coords.xy
        return np.transpose(
            np.array([np.array(list(x_array)), np.array(list(y_array))])
        )


def get_extract_shoreline_extraction_area_array(
    shoreline_extraction_area: gpd.GeoDataFrame,
    output_epsg: int,
    roi_gdf: gpd.GeoDataFrame,
) -> np.ndarray:
    """Extract the shoreline extraction area as a numpy array.

    Args:
        shoreline_extraction_area (GeoDataFrame): The shoreline extraction area.
        output_epsg (int): EPSG code for the output coordinate reference system.
        roi_gdf (GeoDataFrame): The region of interest polygon.

    Returns:
        list: The shoreline extraction area as a list of numpy arrays
    """
    shoreline_extraction_area_array = []
    if shoreline_extraction_area is not None and not roi_gdf.empty:
        # Ensure the extraction area is in the correct CRS
        shoreline_extraction_area_gdf = shoreline_extraction_area.to_crs(
            f"epsg:{output_epsg}"
        )
        roi_gdf = roi_gdf.to_crs(f"epsg:{output_epsg}")

        # Clip the shoreline extraction area to the region of interest
        clipped_shoreline_extraction_area_gdf = shoreline_extraction_area_gdf.clip(
            roi_gdf
        )
        if not clipped_shoreline_extraction_area_gdf.empty:
            shoreline_extraction_area_array = convert_gdf_to_array(
                clipped_shoreline_extraction_area_gdf
            )

    return shoreline_extraction_area_array


def filter_shoreline(
    shoreline,
    shoreline_extraction_area,
    output_epsg,
):
    """Filter the shoreline based on the extraction area.

    Args:
        shoreline (array): The original shoreline data.
        shoreline_extraction_area (GeoDataFrame): The area to extract the shoreline from.
        shoreline_extraction_area (GeoDataFrame): The area to extract the shoreline from.

    Returns:
        np.array: The filtered shoreline as a numpy array of shape (n,2).
    """
    if shoreline_extraction_area is not None:
        # Ensure both the shoreline and extraction area are in the same CRS.
        shoreline_extraction_area_gdf = shoreline_extraction_area.to_crs(
            f"epsg:{output_epsg}"
        )

        # Convert the shoreline to a GeoDataFrame.
        shoreline_gdf = create_gdf_from_type(
            shoreline,
            "lines",
            crs=f"epsg:{output_epsg}",
        )
        if shoreline_gdf is None:
            return shoreline

        # Filter shorelines within the extraction area.
        filtered_shoreline_gdf = ref_poly_filter(
            shoreline_extraction_area_gdf, shoreline_gdf
        )

        # Convert the filtered shoreline back to a numpy array.
        shoreline = convert_shoreline_to_array(filtered_shoreline_gdf)

    return shoreline


def extract_and_filter_shoreline(
    shoreline_extraction_area,
    shoreline,
    satname,
    sl_data,
    acc_georef,
    cloud_cover,
    output_epsg,
    roi_gdf,
):
    """Extract and filter the shoreline based on the extraction area.

    Args:
        shoreline_extraction_area (GeoDataFrame): The area to extract the shoreline from.
        shoreline (array): The original shoreline data.
        metadata (dict): Metadata associated with the satellite imagery.
        satname (str): Name of the satellite.
        sl_data (GeoDataFrame): The satellite image data.
        cloud_cover (float): Cloud cover percentage.
        output_epsg (int): EPSG code for the output coordinate reference system.
        roi_gdf (GeoDataFrame): The region of interest polygon.

    Returns:
        np.array: The filtered shoreline as a numpy array of shape (n,2).
        np.array: The shoreline extraction area as a numpy array of shape (n,2).
    """
    shoreline_extraction_area_array = []

    if shoreline_extraction_area is not None:
        # Ensure both the shoreline and extraction area are in the same CRS.

        shoreline_extraction_area_gdf = shoreline_extraction_area.to_crs(
            f"epsg:{output_epsg}"
        )

        # Convert the shoreline to a GeoDataFrame.
        shoreline_gdf = create_gdf(
            shoreline,
            sl_data,
            satname,
            acc_georef,
            cloud_cover,
            0,
            "lines",
            crs=f"epsg:{output_epsg}",
        )
        if shoreline_gdf is None:
            return shoreline, shoreline_extraction_area_array
        shoreline_gdf.reset_index(drop=True, inplace=True)

        # Filter shorelines within the extraction area.(unclipped version)
        filtered_shoreline_gdf = ref_poly_filter(
            shoreline_extraction_area_gdf, shoreline_gdf
        )

        # Convert the filtered shoreline back to a numpy array.
        shoreline = convert_shoreline_to_array(
            filtered_shoreline_gdf,
        )

        # Convert the clipped extraction area into a numpy array(used later to plot the extraction area on the detection figure)
        # clip the shoreline extraction area to the region of interest
        if roi_gdf.empty:
            raise ValueError(
                "Region of interest is empty. Cannot clip shoreline extraction area."
            )

        roi_gdf = roi_gdf.to_crs(f"epsg:{output_epsg}")
        clipped_shoreline_extraction_area_gdf = shoreline_extraction_area_gdf.clip(
            roi_gdf
        )

        if not clipped_shoreline_extraction_area_gdf.empty:
            shoreline_extraction_area_array = convert_gdf_to_array(
                clipped_shoreline_extraction_area_gdf
            )

    return shoreline, shoreline_extraction_area_array


def get_finite_data(data) -> np.ndarray:
    """
    Extracts the valid (non-NaN) values from the input data array.

    Parameters:
    data (np.ndarray): The input data array.

    Returns:
    np.ndarray: The array containing only the valid values.

    Raises:
    ValueError: If no finite data is available for thresholding.
    """
    valid_mask = np.isfinite(data)  # Create a mask of valid (non-NaN) values
    valid_data = data[valid_mask]  # Extract only the valid values
    if len(valid_data) == 0:
        raise ValueError(
            "Not enough valid pixels found in reference shoreline buffer to extract a shoreline."
        )
    return valid_data


def create_gdf(
    shoreline: List[List[float]],
    date: datetime,
    satname: str,
    geoaccuracy: Union[float, str],
    cloud_cover: float,
    idx: int,
    geomtype: str,
    crs: str = None,
):
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
        if crs:
            gdf.crs = crs
        return gdf

    return None


def create_gdf_from_type(
    shoreline: List[List[float]],
    geomtype: str,
    crs: str = None,
):
    """
    Creates a GeoDataFrame for a given shoreline and its attributes.

    Parameters:
    -----------
    shoreline: List[List[float]]
        List of shoreline coordinates.
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
        gdf = gpd.GeoDataFrame(
            geometry=[geom],
        )
        if crs:
            gdf.crs = crs
        return gdf

    return None


def ref_poly_filter(
    ref_poly_gdf: gpd.GeoDataFrame, raw_shorelines_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Filters shorelines that are within a reference polygon.
    filters extracted shorelines that are not contained within a reference region/polygon

    Args:
        ref_poly_gdf (GeoDataFrame): A GeoDataFrame representing the reference polygon.
        raw_shorelines_gdf (GeoDataFrame): A GeoDataFrame representing the raw shorelines.

    Returns:
        GeoDataFrame: A filtered GeoDataFrame containing shorelines that are within the reference polygon.
    """
    if ref_poly_gdf.empty:
        return raw_shorelines_gdf

    ref_polygon = (
        ref_poly_gdf.geometry.unary_union
    )  # Combine all polygons into a single MultiPolygon if needed

    # Filter out lines that do not intersect the reference polygon
    raw_shorelines_gdf = raw_shorelines_gdf[raw_shorelines_gdf.intersects(ref_polygon)]

    def filter_points_within_polygon(line):
        line_arr = LineString_to_arr(line)
        bool_vals = [ref_polygon.contains(Point(point)) for point in line_arr]
        new_line_arr = line_arr[bool_vals]
        if len(new_line_arr) < 3:
            return None
        return arr_to_LineString(new_line_arr)

    raw_shorelines_gdf.loc[:, "geometry"] = raw_shorelines_gdf["geometry"].apply(
        filter_points_within_polygon
    )
    raw_shorelines_gdf = raw_shorelines_gdf.dropna(subset=["geometry"])

    return raw_shorelines_gdf


def get_pixel_size_for_satellite(satname: str) -> int:
    """Returns the pixel size of a given satellite.
    ["L5", "L7", "L8", "L9"] = 15 meters
    "S2" = 10 meters
    "S1" = 10 meters for VH

    Args:
        satname (str): A string indicating the name of the satellite.

    Returns:
        int: The pixel size of the satellite in meters.

    Raises:
        None.
    """
    pixel_size = 15
    if satname in ["L5", "L7", "L8", "L9"]:
        pixel_size = 15
    elif satname == "S2":
        pixel_size = 10
    elif satname == "S1":
        pixel_size = 10
    return pixel_size


# Main function for batch shoreline detection
def extract_shorelines(
    metadata,
    settings,
    output_directory: str = None,
    shoreline_extraction_area: gpd.GeoDataFrame = None,
):
    """
    Main function to extract shorelines from satellite images

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
            are erroneously being masked on the image
        'min_beach_area': int
            minimum allowable object area (in metres^2) for the class 'sand',
            the area is converted to number of connected pixels
        'min_length_sl': int
            minimum length (in metres) of shoreline contour to be valid
        'sand_color': str
            default', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline
        'adjust_detection': bool
            if True, allows user to manually adjust the detected shoreline
        'pan_off': bool
            if True, no pan-sharpening is performed on Landsat 7,8 and 9 imagery
        's2cloudless_prob': float [0,100)
            threshold to identify cloud pixels in the s2cloudless probability mask
        'sar_save_probability': bool (default: False)
            if True, saves P(water) of every model-segmented S1 scene as a GeoTIFF
            in S1/prob/ (see save_sar_probability)
    output_directory: str (default: None)
        The directory to save the output files. If None, the output files will be saved in the same directory as the input files.
    shoreline_extraction_area: gpd.GeoDataFrame (default: None)
        A geodataframe containing polygons indicating the areas to extract the shoreline from. Any shoreline outside of these polygons will be discarded.

    Returns:
    -----------
    output: dict
        contains the extracted shorelines and corresponding dates + metadata

    """
    try:
        import importlib.resources

        filepath_models = os.path.abspath(importlib.resources.files(models))
    except AttributeError:
        from importlib_resources import files

        filepath_models = os.path.abspath(files(models))

    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    collection = settings["inputs"]["landsat_collection"]

    sitename_location = os.path.join(filepath_data, sitename)
    # set up logger at the output directory if it is provided otherwise set up logger at the sitename location
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        logger = setup_logger(
            output_directory,
            "extract_shorelines_report",
            log_format="%(levelname)s - %(message)s",
        )
    else:
        logger = setup_logger(
            sitename_location,
            "extract_shorelines_report",
            log_format="%(levelname)s - %(message)s",
        )

    logger.info(f"Please read the following information carefully:\n")
    logger.info(
        "find_wl_contours2: A method for extracting shorelines that uses the sand water interface detected with the model to refine the threshold that's used to detect shorelines .\n  - This is the default method used when there are enough sand pixels within the reference shoreline buffer.\n"
    )
    logger.info(
        "find_wl_contours1: This shoreline extraction method uses a threshold to differentiate between water and land pixels in images, relying on Modified Normalized Difference Water Index (MNDWI) values. However, it may inaccurately classify snow and ice as water, posing a limitation in certain environments.\n  - This is only used when not enough sand pixels are detected within the reference shoreline buffer.\n"
    )
    logger.info(
        "---------------------------------------------------------------------------------------------------------------------"
    )
    # initialise output structure
    output = dict([])
    # create a subfolder to store the .jpg images showing the detection
    filepath_jpg = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    if not os.path.exists(filepath_jpg):
        os.makedirs(filepath_jpg)
    # close all open figures
    plt.close("all")
    output_epsg = settings["output_epsg"]
    default_min_length_sl = settings["min_length_sl"]
    # loop through satellite list
    for satname in metadata.keys():
        # get images
        filepath = SDS_tools.get_filepath(settings["inputs"], satname)
        filenames = metadata[satname]["filenames"]

        # initialise the output variables
        output_timestamp = []  # datetime at which the image was acquired (UTC time)
        output_shoreline = []  # vector of shoreline points
        output_filename = (
            []
        )  # filename of the images from which the shorelines where derived
        output_cloudcover = []  # cloud cover of the images
        output_geoaccuracy = []  # georeferencing accuracy of the images
        output_idxkeep = (
            []
        )  # index that were kept during the analysis (cloudy images are skipped)
        output_t_mndwi = []  # MNDWI threshold used to map the shoreline
        output_segmentation = []  # how each shoreline was derived, and so what
        # output_t_mndwi holds for it (see SDS_tools.SEGMENTATION_*)
        output_water_fraction = []  # water fraction in the reference buffer (SAR
        # only, NaN for optical) - the SAR transect QC quantity

        # load classifiers

        # get the pixel size for the satellite eg. 15m for Landsat 5,7,8,9 and 10m for Sentinel 2
        pixel_size = get_pixel_size_for_satellite(satname)

        clf = load_classifier(
            satname=satname,
            sand_color=settings["sand_color"],
            model_path=filepath_models,
            sklearn_version=sklearn.__version__,
        )

        # load the SAR segmentation model once for the whole satellite, not per scene:
        # the .onnx is ~130 MB. None means every S1 scene is thresholded with Otsu.
        sar_segmenter = (
            load_sar_segmenter(settings, logger=logger) if satname == "S1" else None
        )
        if satname == "S1":
            # a typo here would silently look like the default, so say so once
            warn_unknown_sar_detection_polarization(settings, logger=logger)

        # convert settings['min_beach_area'] from metres to pixels
        min_beach_area_pixels = np.ceil(settings["min_beach_area"] / pixel_size**2)

        # reduce min shoreline length for L7 because of the diagonal bands
        settings["min_length_sl"] = 200 if satname == "L7" else default_min_length_sl
        # set the minimum shoreline length for L7 to 200m due to the diagonal bands
        if satname == "L7":
            logger.info(
                f"WARNING: CoastSat has hard-coded the value for the minimum shoreline length for L7 to 200\n\n"
            )
        logger.info(
            f"Extracting shorelines for {satname} Minimum Shoreline Length: {settings['min_length_sl']}\n\n"
        )

        # loop through the images
        for i in tqdm(
            range(len(filenames)),
            desc=f"{satname}: Mapping Shorelines",
            leave=True,
            position=0,
        ):
            # get image filename
            fn = SDS_tools.get_filenames(filenames[i], filepath, satname)
            shoreline_date = os.path.basename(fn[0])[:19]

            try:
                im_ms, georef, cloud_mask, im_nodata = preprocess_image(
                    fn=fn,
                    satname=satname,
                    settings=settings,
                    collection=collection,
                )
            except SDS_preprocess.SkipImageError as e:
                message = (
                    f"{satname} {shoreline_date}: Skipped during preprocessing. {e}"
                )
                logger.warning(message)
                continue
            except FileNotFoundError as e:
                logger.error(
                    f"Could not extract shoreline for {shoreline_date} due to missing files.{e}"
                )
                continue

            # get image spatial reference system (epsg code) from metadata dict
            image_epsg = metadata[satname]["epsg"][i]

            try:
                cloud_cover_combined, cloud_cover, cloud_mask_adv = (
                    compute_cloud_metrics(cloud_mask=cloud_mask, im_nodata=im_nodata)
                )
            except ValueError as e:
                logger.error(
                    f"{satname} {shoreline_date}: Skipped due to invalid no-data mask. "
                    f"Reason: {e}. Image file(s): {fn}"
                )
                print(
                    f"\n{satname} {shoreline_date}: Skipped shoreline extraction due to invalid no-data mask ({e}). "
                    f"Image file(s): {fn}"
                )
                continue

            if should_skip_image(
                cloud_cover_combined=cloud_cover_combined,
                cloud_cover=cloud_cover,
                settings=settings,
                logger=logger,
                satname=satname,
                date_str=shoreline_date,
            ):
                continue

            logger.info(f"{satname} {shoreline_date} cloud cover : {cloud_cover:.2%}")

            # calculate a buffer around the reference shoreline (if any has been digitised)
            im_ref_buffer = create_shoreline_buffer(
                cloud_mask.shape, georef, image_epsg, pixel_size, settings
            )

            # for SAR the figure shows a single band, not the multi-channel im_ms
            im_display = im_ms
            # which class each value of im_labels is, for the figure legend
            class_labels = ("water", "land")
            segmentation_method = SDS_tools.SEGMENTATION_MNDWI
            # fraction of water in the reference shoreline buffer; only computed for
            # SAR scenes, where it is the QC quantity (see SDS_transects.sar_qc_keep)
            water_fraction = np.nan

            if satname == "S1":
                # mean of VV and VH for a dual-polarization scene, the single band
                # otherwise (see get_sar_otsu_image). Used as the grayscale backdrop of
                # the detection figure whichever method segmented the scene, so the two
                # are directly comparable. The SAR imagery is deliberately not smoothed
                # or speckle-filtered.
                im_display = get_sar_otsu_image(im_ms, fn, settings)

                segmentation = segment_sar_water(
                    im_ms,
                    fn,
                    sar_segmenter,
                    settings,
                    logger=logger,
                    date_str=shoreline_date,
                )

                if segmentation is not None:
                    im_water, valid, im_prob = segmentation
                    logger.info(
                        f"{satname} {shoreline_date}: segmented "
                        f"{np.sum(im_water) / im_water.size:.2%} water"
                    )
                    im_labels = im_water
                    class_labels = ("land", "water")
                    segmentation_method = SDS_tools.SEGMENTATION_SAR_MODEL
                    water_fraction = compute_sar_water_fraction(
                        im_water, im_ref_buffer & valid
                    )
                    if settings.get("sar_save_probability", False):
                        save_sar_probability(
                            im_prob, valid, fn, logger=logger, date_str=shoreline_date
                        )
                    # invalid pixels must not generate contours of their own
                    shoreline = find_shoreline_SAR(
                        im_water, im_ref_buffer & valid, georef, image_epsg, settings
                    )
                    # there is no dB threshold to report for a segmentation model, and
                    # storing the probability cutoff here would be read as one
                    t_mndwi = np.nan
                else:
                    segmentation_method = SDS_tools.SEGMENTATION_SAR_OTSU
                    logger.info(
                        f"{satname} {shoreline_date}: thresholding "
                        f"{describe_sar_otsu_image(fn, settings)}"
                    )

                    # instead of applying the classifier here, we will apply the otsu
                    # threshold. min_beach_area_pixels, not settings['min_beach_area']:
                    # the latter is in m^2, which at 10 m/pixel would drop objects 100x
                    # larger than asked for (the optical path converts the same way)
                    otsu_image, threshold = otsu_threshold(
                        im_display, min_beach_area_pixels
                    )
                    shoreline = find_shoreline_SAR(
                        otsu_image, im_ref_buffer, georef, image_epsg, settings
                    )
                    im_labels = otsu_image
                    # Otsu marks the high-backscatter class - land - as True
                    water_fraction = compute_sar_water_fraction(
                        ~otsu_image, im_ref_buffer
                    )
                    # convert from list to number if list
                    t_mndwi = (
                        threshold[0]
                        if isinstance(threshold, (list, np.ndarray))
                        else threshold
                    )

            else:
                # classify image in 4 classes (sand, whitewater, water, other) with NN classifier
                im_classif, im_labels = classify_image_NN(
                    im_ms, cloud_mask, min_beach_area_pixels, clf
                )

                # sand, whitewater, water, other
                class_mapping = {
                    0: "sand",
                    1: "whitewater",
                    2: "water",
                }

                logger.info(
                    f"{satname} {shoreline_date}: "
                    + f" ,".join(
                        f"{class_name}: {np.sum(im_labels[:, :, index])/im_labels[:, :, index].size:.2%}"
                        for index, class_name in class_mapping.items()
                    )
                )

                # if adjust_detection is True, let the user adjust the detected shoreline
                if settings["adjust_detection"]:
                    date = filenames[i][:19]
                    skip_image, shoreline, t_mndwi = adjust_detection(
                        im_ms,
                        cloud_mask,
                        im_nodata,
                        im_labels,
                        im_ref_buffer,
                        image_epsg,
                        georef,
                        settings,
                        date,
                        satname,
                    )
                    # if the user decides to skip the image, continue and do not save the mapped shoreline
                    if skip_image:
                        continue
                # otherwise map the contours automatically with one of the two following functions:
                # if there are pixels in the 'sand' class --> use find_wl_contours2 (enhanced)
                # otherwise use find_wl_contours1 (traditional)

                try:  # use try/except structure for long runs
                    shoreline, t_mndwi = find_shoreline_optical(
                        im_ms,
                        im_labels,
                        im_ref_buffer,
                        cloud_mask,
                        cloud_mask_adv,
                        im_nodata,
                        georef,
                        image_epsg,
                        satname,
                        shoreline_date,
                        settings,
                        logger=logger,
                    )
                except Exception as e:
                    print(
                        f"{satname} {shoreline_date}: Could not map shoreline due to error {str(e)}"
                    )
                    logger.error(
                        f"{satname} {shoreline_date}: Could not map shoreline due to error {e}\n{traceback.format_exc()}"
                    )
                    continue

            # convert the polygon coordinates of ROI to gdf
            height, width = im_ms.shape[:2]
            output_epsg = settings["output_epsg"]
            date = filenames[i][:19]
            roi_gdf = SDS_preprocess.create_gdf_from_image_extent(
                height, width, georef, image_epsg, output_epsg
            )

            # filter shorelines within the extraction area
            shoreline = filter_shoreline(
                shoreline, shoreline_extraction_area, output_epsg
            )
            shoreline_extraction_area_array = (
                get_extract_shoreline_extraction_area_array(
                    shoreline_extraction_area, output_epsg, roi_gdf
                )
            )

            # visualize the mapped shorelines, there are two options:
            # if settings['check_detection'] = True, shows the detection to the user for accept/reject
            # if settings['save_figure'] = True, saves a figure for each mapped shoreline
            if settings["check_detection"] or settings["save_figure"]:
                date = filenames[i][:19]
                if not settings["check_detection"]:
                    plt.ioff()  # turning interactive plotting off
                from coastsat.SDS_plotting import plot_detection

                skip_image = plot_detection(
                    im_display,
                    im_labels,
                    shoreline,
                    image_epsg,
                    georef,
                    settings,
                    date,
                    satname,
                    im_ref_buffer=im_ref_buffer,
                    output_directory=output_directory,
                    cloud_mask=cloud_mask,
                    shoreline_extraction_area=shoreline_extraction_area_array,
                    is_sar=True if satname == "S1" else False,
                    class_labels=class_labels,
                )

                # if the user decides to skip the image, continue and do not save the mapped shoreline
                if skip_image:
                    continue

            # append to output variables
            output_timestamp.append(metadata[satname]["dates"][i])
            output_shoreline.append(shoreline)
            output_filename.append(filenames[i])
            output_cloudcover.append(cloud_cover)
            output_geoaccuracy.append(metadata[satname]["acc_georef"][i])
            output_idxkeep.append(i)
            output_t_mndwi.append(t_mndwi)
            output_segmentation.append(segmentation_method)
            output_water_fraction.append(water_fraction)

        # create dictionnary of output
        output[satname] = {
            "dates": output_timestamp,
            "shorelines": output_shoreline,
            "filename": output_filename,
            "cloud_cover": output_cloudcover,
            "geoaccuracy": output_geoaccuracy,
            "idx": output_idxkeep,
            "MNDWI_threshold": output_t_mndwi,
            "segmentation_method": output_segmentation,
            # appended for every satellite (NaN for optical): merge_output builds its
            # key list from the first satellite only, so a column missing on one
            # satellite would be dropped for all of them
            "water_fraction": output_water_fraction,
        }

    # close figure window if still open
    if plt.get_fignums():
        plt.close()

    # change the format to have one list sorted by date with all the shorelines (easier to use)
    output = SDS_tools.merge_output(output)

    # save putput structure as output.json
    # don't do this for now
    # if output_directory is not None:
    #     filepath = output_directory

    filepath = os.path.join(filepath_data, sitename)
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    json_path = os.path.join(filepath, sitename + "_output.json")
    SDS_preprocess.write_to_json(json_path, output)
    # release the logger as it is no longer needed
    release_logger(logger)

    return output


###################################################################################################
# IMAGE CLASSIFICATION FUNCTIONS
###################################################################################################


def calculate_features(im_ms, cloud_mask, im_bool):
    """
    Calculates features on the image that are used for the supervised classification.
    The features include spectral normalized-difference indices and standard
    deviation of the image for all the bands and indices.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_bool: np.array
        2D array of boolean indicating where on the image to calculate the features

    Returns:
    -----------
    features: np.array
        matrix containing each feature (columns) calculated for all
        the pixels (rows) indicated in im_bool

    """

    # add all the multispectral bands
    features = np.expand_dims(im_ms[im_bool, 0], axis=1)
    for k in range(1, im_ms.shape[2]):
        feature = np.expand_dims(im_ms[im_bool, k], axis=1)
        features = np.append(features, feature, axis=-1)
    # NIR-G
    im_NIRG = SDS_tools.nd_index(im_ms[:, :, 3], im_ms[:, :, 1], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRG[im_bool], axis=1), axis=-1)
    # SWIR-G
    im_SWIRG = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRG[im_bool], axis=1), axis=-1)
    # NIR-R
    im_NIRR = SDS_tools.nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)
    features = np.append(features, np.expand_dims(im_NIRR[im_bool], axis=1), axis=-1)
    # SWIR-NIR
    im_SWIRNIR = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 3], cloud_mask)
    features = np.append(features, np.expand_dims(im_SWIRNIR[im_bool], axis=1), axis=-1)
    # B-R
    im_BR = SDS_tools.nd_index(im_ms[:, :, 0], im_ms[:, :, 2], cloud_mask)
    features = np.append(features, np.expand_dims(im_BR[im_bool], axis=1), axis=-1)
    # calculate standard deviation of individual bands
    for k in range(im_ms.shape[2]):
        im_std = SDS_tools.image_std(im_ms[:, :, k], 1)
        features = np.append(features, np.expand_dims(im_std[im_bool], axis=1), axis=-1)
    # calculate standard deviation of the spectral indices
    im_std = SDS_tools.image_std(im_NIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool], axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRG, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool], axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_NIRR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool], axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_SWIRNIR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool], axis=1), axis=-1)
    im_std = SDS_tools.image_std(im_BR, 1)
    features = np.append(features, np.expand_dims(im_std[im_bool], axis=1), axis=-1)

    return features


def classify_image_NN(im_ms, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in the image in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        Pansharpened RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        pre-trained classifier

    Returns:
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)

    """

    # calculate features
    vec_features = calculate_features(
        im_ms, cloud_mask, np.ones(cloud_mask.shape).astype(bool)
    )
    vec_features[np.isnan(vec_features)] = (
        1e-9  # NaN values are create when std is too close to 0
    )

    # remove NaNs and cloudy pixels
    vec_cloud = cloud_mask.reshape(cloud_mask.shape[0] * cloud_mask.shape[1])
    vec_nan = np.any(np.isnan(vec_features), axis=1)
    vec_inf = np.any(np.isinf(vec_features), axis=1)
    vec_mask = np.logical_or(vec_cloud, np.logical_or(vec_nan, vec_inf))
    vec_features = vec_features[~vec_mask, :]

    # classify pixels
    labels = clf.predict(vec_features)

    # recompose image
    vec_classif = np.nan * np.ones((cloud_mask.shape[0] * cloud_mask.shape[1]))
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape((cloud_mask.shape[0], cloud_mask.shape[1]))

    # create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3
    # remove small patches of sand or water that could be around the image (usually noise)
    im_sand = morphology.remove_small_objects(
        im_sand, min_size=min_beach_area, connectivity=2
    )
    im_water = morphology.remove_small_objects(
        im_water, min_size=min_beach_area, connectivity=2
    )

    im_labels = np.stack((im_sand, im_swash, im_water), axis=-1)

    return im_classif, im_labels


def classify_image_NN_sar(im_sar, cloud_mask, min_beach_area, clf):
    """
    Classifies every pixel in SAR imagery in one of 4 classes:
        - sand                                          --> label = 1
        - whitewater (breaking waves and swash)         --> label = 2
        - water                                         --> label = 3
        - other (vegetation, buildings, rocks...)       --> label = 0

    The classifier is a Neural Network that is already trained.

    Arguments:
    -----------
    im_sar: np.array
        SAR image with shape (x, y, 1)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    min_beach_area: int
        Minimum number of pixels that have to be connected to belong to the SAND class
    clf: joblib object
        Pre-trained classifier

    Returns:
    -----------
    im_classif: np.array
        2D image containing labels
    im_labels: np.array of booleans
        3D image containing a boolean image for each class (im_classif == label)
    """
    # Flatten the SAR image and cloud mask
    vec_sar = im_sar.reshape(-1, 3)  # Reshape to (num_pixels, 3)
    vec_cloud = cloud_mask.flatten()

    # Remove cloudy pixels
    vec_mask = vec_cloud
    vec_sar = vec_sar[~vec_mask, :]

    # Handle NaN or infinite values
    vec_sar[np.isnan(vec_sar)] = 1e-9
    vec_sar[np.isinf(vec_sar)] = 1e-9

    # Classify pixels
    labels = clf.predict(vec_sar)

    # Recompose the classified image
    vec_classif = np.nan * np.ones(im_sar.shape[0] * im_sar.shape[1])
    vec_classif[~vec_mask] = labels
    im_classif = vec_classif.reshape(im_sar.shape[:2])

    # Create a stack of boolean images for each label
    im_sand = im_classif == 1
    im_swash = im_classif == 2
    im_water = im_classif == 3

    # Remove small patches of sand or water that could be noise
    im_sand = morphology.remove_small_objects(
        im_sand, min_size=min_beach_area, connectivity=2
    )
    im_water = morphology.remove_small_objects(
        im_water, min_size=min_beach_area, connectivity=2
    )

    im_labels = np.stack((im_sand, im_swash, im_water), axis=-1)

    return im_classif, im_labels


###################################################################################################
# CONTOUR MAPPING FUNCTIONS
###################################################################################################


def find_wl_contours1(im_ndwi, cloud_mask, im_ref_buffer):
    """
    Traditional method for shoreline detection using a global threshold.
    Finds the water line by thresholding the Normalized Difference Water Index
    and applying the Marching Squares Algorithm to contour the iso-value
    corresponding to the threshold.

    KV WRL 2018

    Arguments:
    -----------
    im_ndwi: np.ndarray
        Image (2D) with the NDWI (water index)
    cloud_mask: np.ndarray
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline

    Returns:
    -----------
    contours: list of np.arrays
        contains the coordinates of the contour lines
    t_mwi: float
        Otsu threshold used to map the contours

    """
    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # create a buffer around the reference shoreline and reshape it into a vector eg. shape (nrows*ncols,)
    vec_buffer = im_ref_buffer.reshape(nrows * ncols)

    # reshape spectral index image to vector eg. shape (nrows*ncols,)
    vec_ndwi = im_ndwi.reshape(nrows * ncols)
    vec_mask = cloud_mask.reshape(nrows * ncols)

    # keep pixels that are in the buffer and not in the cloud mask
    vec = vec_ndwi[np.logical_and(vec_buffer, ~vec_mask)]
    vec = vec[~np.isnan(vec)]
    if len(vec) == 0:
        raise ValueError(
            "Not enough valid pixels found in reference shoreline buffer to extract a shoreline."
        )
    # apply otsu's thresholding method to find the optimal threshold separating water and land pixels
    t_otsu = filters.threshold_otsu(vec)
    # use Marching Squares algorithm to detect contours on ndwi image
    im_ndwi_buffer = np.copy(im_ndwi)
    im_ndwi_buffer[~im_ref_buffer] = np.nan
    contours = measure.find_contours(im_ndwi_buffer, t_otsu)
    # remove contours that contain NaNs (due to cloud pixels in the contour)
    contours = process_contours(contours)

    return contours, t_otsu


def find_wl_contours2(im_ms, im_labels, cloud_mask, im_ref_buffer):
    """
    New robust method for extracting shorelines. Incorporates the classification
    component to refine the treshold and make it specific to the sand/water interface.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_ref_buffer: np.array
        binary image containing a buffer around the reference shoreline

    Returns:
    -----------
    contours_mwi: list of np.arrays
        contains the coordinates of the contour lines extracted from the
        MNDWI (Modified Normalized Difference Water Index) image
    t_mwi: float
        Otsu sand/water threshold used to map the contours

    """

    nrows = cloud_mask.shape[0]
    ncols = cloud_mask.shape[1]

    # calculate Normalized Difference Modified Water Index (SWIR - G)
    im_mwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    # calculate Normalized Difference Modified Water Index (NIR - G)
    im_wi = SDS_tools.nd_index(im_ms[:, :, 3], im_ms[:, :, 1], cloud_mask)
    # stack indices together
    im_ind = np.stack((im_wi, im_mwi), axis=-1)
    vec_ind = im_ind.reshape(nrows * ncols, 2)

    # reshape labels into vectors
    vec_sand = im_labels[:, :, 0].reshape(ncols * nrows)
    vec_water = im_labels[:, :, 2].reshape(ncols * nrows)

    # Make a buffer around the reference shoreline and reshape it into a vector
    vec_buffer = im_ref_buffer.reshape(nrows * ncols)

    # select water/sand pixels that are within the buffer
    int_water = vec_ind[np.logical_and(vec_buffer, vec_water), :]
    int_sand = vec_ind[np.logical_and(vec_buffer, vec_sand), :]

    # make sure both classes have the same number of pixels before thresholding
    if len(int_water) > 0 and len(int_sand) > 0:
        if np.argmin([int_sand.shape[0], int_water.shape[0]]) == 1:
            int_sand = int_sand[
                np.random.choice(int_sand.shape[0], int_water.shape[0], replace=False),
                :,
            ]
        else:
            int_water = int_water[
                np.random.choice(int_water.shape[0], int_sand.shape[0], replace=False),
                :,
            ]

    # threshold the sand/water intensities
    int_all = np.append(int_water, int_sand, axis=0)

    # if only no data pixels and cloud pixel (indicated by NaN) are found in the buffer, raise an error
    valid_mwi = get_finite_data(int_all[:, 0])
    valid_wi = get_finite_data(int_all[:, 1])

    t_mwi = filters.threshold_otsu(valid_mwi)
    t_wi = filters.threshold_otsu(valid_wi)

    # find contour with Marching-Squares algorithm
    im_wi_buffer = np.copy(im_wi)
    im_wi_buffer[~im_ref_buffer] = np.nan
    im_mwi_buffer = np.copy(im_mwi)
    im_mwi_buffer[~im_ref_buffer] = np.nan
    contours_wi = measure.find_contours(im_wi_buffer, t_wi)
    contours_mwi = measure.find_contours(im_mwi_buffer, t_mwi)
    # remove contour points that are NaNs (around clouds)
    contours_wi = process_contours(contours_wi)
    contours_mwi = process_contours(contours_mwi)

    # only return MNDWI contours and threshold
    return contours_mwi, t_mwi


###################################################################################################
# SHORELINE PROCESSING FUNCTIONS
###################################################################################################


def densify_line_coords(coords: np.array, max_spacing: float) -> np.array:
    """
    Densify a 2D LineString so no segment is longer than max_spacing.
    coords must be in a projected CRS where units are meters.

    Example: Make a sparse line segment consisting of two points 100m apart into a line segment with points every 10m.
    The Linestring looks the same but has more points

    Arguments:
    coords: np.array
        array of shape (n, 2) containing the coordinates of the line to be densified
    max_spacing: float
         maximum allowed spacing between points in the output coordinates

    Returns:
    np.array
        array of shape (m, 2) containing the densified coordinates
    """
    line = LineString(coords)

    if line.is_empty or line.length == 0:
        return coords

    # Shapely >= 2.0
    if hasattr(line, "segmentize"):
        return np.asarray(line.segmentize(max_spacing).coords)

    # Fallback for older Shapely
    distances = np.arange(0, line.length, max_spacing)
    if len(distances) == 0 or distances[-1] < line.length:
        distances = np.append(distances, line.length)

    pts = [line.interpolate(d) for d in distances]
    return np.array([[p.x, p.y] for p in pts])


def create_shoreline_buffer(im_shape, georef, image_epsg, pixel_size, settings):
    """
    Creates a buffer around the reference shoreline. The size of the buffer is
    given by settings['max_dist_ref'].

    KV WRL 2018

    Arguments:
    -----------
    im_shape: np.array
        size of the image (rows,columns)
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    pixel_size: int
        size of the pixel in metres (15 for Landsat, 10 for Sentinel-2)
    settings: dict with the following keys
        'output_epsg': int
            output spatial reference system
        'reference_shoreline': np.array
            coordinates of the reference shoreline
        'max_dist_ref': int
            maximum distance from the reference shoreline in metres

    Returns:
    -----------
    im_buffer: np.array
        binary image, True where the buffer is, False otherwise

    """
    # initialise the image buffer
    im_buffer = np.ones(im_shape).astype(bool)

    if "reference_shoreline" in settings.keys():
        ref_sl = settings["reference_shoreline"][:, :2]

        # Convert reference shoreline to the image CRS first
        ref_sl_conv = SDS_tools.convert_epsg(
            ref_sl, settings["output_epsg"], image_epsg
        )[:, :2]

        # Densify before converting to pixels.
        # Use half a pixel so the reference line is safely sampled.
        densify_spacing = settings.get("ref_sl_densify_spacing", pixel_size / 2)
        ref_sl_conv = densify_line_coords(ref_sl_conv, densify_spacing)

        # Convert dense line to pixel coordinates
        ref_sl_pix = SDS_tools.convert_world2pix(ref_sl_conv, georef)
        ref_sl_pix_rounded = np.round(ref_sl_pix).astype(int)

        # Rasterize the dense line
        im_binary = np.zeros(im_shape, dtype=bool)

        for j in range(len(ref_sl_pix_rounded) - 1):
            rr, cc = skdraw.line(
                ref_sl_pix_rounded[j, 1],
                ref_sl_pix_rounded[j, 0],
                ref_sl_pix_rounded[j + 1, 1],
                ref_sl_pix_rounded[j + 1, 0],
            )

            # Clip line pixels to image bounds
            valid = (rr >= 0) & (rr < im_shape[0]) & (cc >= 0) & (cc < im_shape[1])

            im_binary[rr[valid], cc[valid]] = True

        # Dilate the binary reference line to create the buffer. This only makes
        # sense when a reference shoreline was given; without one the buffer stays
        # all True, so the whole image is searched.
        max_dist_ref_pixels = int(np.ceil(settings["max_dist_ref"] / pixel_size))
        se = morphology.disk(max_dist_ref_pixels)
        im_buffer = SDS_tools.binary_dilation(im_binary, se)

    return im_buffer


def process_contours(contours):
    """
    Remove contours that contain NaNs, usually these are contours that are in contact
    with clouds.

    KV WRL 2020

    Arguments:
    -----------
    contours: list of np.array
        image contours as detected by the function skimage.measure.find_contours

    Returns:
    -----------
    contours: list of np.array
        processed image contours (only the ones that do not contains NaNs)

    """

    # initialise variable
    contours_nonans = []
    # loop through contours and only keep the ones without NaNs
    for k in range(len(contours)):
        if np.any(np.isnan(contours[k])):
            index_nan = np.where(np.isnan(contours[k]))[0]
            contours_temp = np.delete(contours[k], index_nan, axis=0)
            if len(contours_temp) > 1:
                contours_nonans.append(contours_temp)
        else:
            contours_nonans.append(contours[k])

    return contours_nonans


def process_shoreline(
    contours, cloud_mask, im_nodata, georef, image_epsg, settings, **kwargs
):
    """
    Converts the contours from image coordinates to world coordinates. This function also removes the contours that:
        1. are too small to be a shoreline (based on the parameter settings['min_length_sl'])
        2. are too close to cloud pixels (based on the parameter settings['dist_clouds'])
        3. are adjacent to noData pixels

    KV WRL 2018

    Arguments:
    -----------
        contours: np.array or list of np.array
            image contours as detected by the function find_contours
        cloud_mask: np.array
            2D cloud mask with True where cloud pixels are
        im_nodata: np.array
            2D mask with True where noData pixels are
        georef: np.array
            vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
        image_epsg: int
            spatial reference system of the image from which the contours were extracted
        settings: dict
            contains the following fields:
        output_epsg: int
            output spatial reference system
        min_length_sl: float
            minimum length of shoreline perimeter to be kept (in meters)
        dist_clouds: int
            distance in metres defining a buffer around cloudy pixels where the shoreline cannot be mapped

    Returns:
    -----------
        shoreline: np.array
            array of points with the X and Y coordinates of the shoreline

    """
    logger = kwargs.get("logger", None)
    # convert pixel coordinates to world coordinates
    contours_world = SDS_tools.convert_pix2world(contours, georef)
    # convert world coordinates to desired spatial reference system
    contours_epsg = SDS_tools.convert_epsg(
        contours_world, image_epsg, settings["output_epsg"]
    )

    # 1. Remove contours that have a perimeter < min_length_sl (provided in settings dict)
    # this enables to remove the very small contours that do not correspond to the shoreline
    contours_long = []
    for l, wl in enumerate(contours_epsg):
        coords = [(wl[k, 0], wl[k, 1]) for k in range(len(wl))]
        a = LineString(coords)  # shapely LineString structure
        if a.length >= settings["min_length_sl"]:
            contours_long.append(wl)
    # format points into np.array
    x_points = np.array([])
    y_points = np.array([])
    for k in range(len(contours_long)):
        x_points = np.append(x_points, contours_long[k][:, 0])
        y_points = np.append(y_points, contours_long[k][:, 1])
    contours_array = np.transpose(np.array([x_points, y_points]))

    shoreline = contours_array

    if logger:
        logger.info(
            f"Number of shorelines before removing shorelines < {settings['min_length_sl']}m: {len(contours_epsg)} shorelines. Number of shorelines after filtering shorelines: {len(contours_long)} shorelines"
        )

    if len(shoreline) == 0:
        return shoreline
    if logger:
        logger.info(
            f"Number of shoreline points before removing points within {settings['dist_clouds']}m of cloud mask {len(shoreline)}"
        )
    # 2. Remove any shoreline points that are close to cloud pixels (effect of shadows)
    if np.sum(np.sum(cloud_mask)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(cloud_mask)
        idx_cloud = np.array(
            [(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))]
        )
        # convert to world coordinates and same epsg as the shoreline points
        coords_cloud = SDS_tools.convert_epsg(
            SDS_tools.convert_pix2world(idx_cloud, georef),
            image_epsg,
            settings["output_epsg"],
        )[:, :-1]
        # only keep the shoreline points that are at least 30m from any cloud pixel
        idx_keep = np.ones(len(shoreline)).astype(bool)
        for k in range(len(shoreline)):
            if np.any(
                np.linalg.norm(shoreline[k, :] - coords_cloud, axis=1)
                < settings["dist_clouds"]
            ):
                idx_keep[k] = False
        shoreline = shoreline[idx_keep]
    if logger:
        logger.info(
            f"Number of shoreline points after removing points within {settings['dist_clouds']}m of cloud mask {len(shoreline)}"
        )

    if len(shoreline) == 0:
        return shoreline
    if logger:
        logger.info(
            f"Number of shoreline points before removing points within 30m of no data pixel {len(shoreline)}"
        )
    # 3. Remove any shoreline points that are attached to nodata pixels
    if np.sum(np.sum(im_nodata)) > 0:
        # get the coordinates of the cloud pixels
        idx_cloud = np.where(im_nodata)
        idx_cloud = np.array(
            [(idx_cloud[0][k], idx_cloud[1][k]) for k in range(len(idx_cloud[0]))]
        )
        # convert to world coordinates and same epsg as the shoreline points
        coords_cloud = SDS_tools.convert_epsg(
            SDS_tools.convert_pix2world(idx_cloud, georef),
            image_epsg,
            settings["output_epsg"],
        )[:, :-1]
        # only keep the shoreline points that are at least 30m from any nodata pixel
        idx_keep = np.ones(len(shoreline)).astype(bool)
        for k in range(len(shoreline)):
            if np.any(np.linalg.norm(shoreline[k, :] - coords_cloud, axis=1) < 30):
                idx_keep[k] = False
        shoreline = shoreline[idx_keep]

    if logger:
        logger.info(
            f"Number of shoreline points after removing points within 30m of no data pixel {len(shoreline)}"
        )

    return shoreline


###################################################################################################
# INTERACTIVE/PLOTTING FUNCTIONS
###################################################################################################


# Normalize SAR image for display (simulate .convert("L"))
def normalize_grayscale(image):
    p_low, p_high = np.percentile(image, (1, 99))  # Ignore extreme outliers
    image_clipped = np.clip(image, p_low, p_high)
    return (image_clipped - p_low) / (p_high - p_low)


def show_sar_detection(
    im_ms,
    im_labels,
    shoreline,
    image_epsg,
    georef,
    settings,
    date,
    satname,
    im_ref_buffer=None,
    output_directory: str = None,
):
    """
    Shows the detected shoreline on SAR imagery.

    Arguments:
    -----------
    im_ms: np.array
        2D SAR image
    im_labels: np.array
        2D binary segmentation mask (e.g., from Otsu)
    shoreline: np.array
        array of shoreline coordinates (X, Y)
    image_epsg: int
        EPSG of image CRS
    georef: np.array
        GDAL geotransform
    settings: dict
        Includes saving and directory config
    date: str
        Image capture date
    satname: str
        Satellite name
    im_ref_buffer: np.array (optional)
        Reference shoreline buffer (bool mask)
    output_directory: str (optional)
        Directory to save output
    """
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    filepath = (
        os.path.join(output_directory, "jpg_files", "detection")
        if output_directory
        else os.path.join(filepath_data, sitename, "jpg_files", "detection")
    )
    os.makedirs(filepath, exist_ok=True)

    # Convert shoreline world to pixel coordinates
    try:
        sl_pix = SDS_tools.convert_world2pix(
            SDS_tools.convert_epsg(shoreline, settings["output_epsg"], image_epsg)[
                :, [0, 1]
            ],
            georef,
        )
    except:
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    # Prepare figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # define legend elements
    shoreline_line = mlines.Line2D([], [], color="red", label="Shoreline")
    buffer_patch = mpatches.Patch(
        color="red", alpha=0.4, label="Reference shoreline buffer"
    )
    otsu_patch_1 = mpatches.Patch(color="yellow", label="Otsu class 1")
    otsu_patch_2 = mpatches.Patch(color="blue", label="Otsu class 2")

    # First plot: raw SAR image with shoreline
    im_display = normalize_grayscale(im_ms)
    ax1.imshow(im_display, cmap="gray")  # plot the SAR image
    ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "r.", markersize=1)  # plot the shoreline
    ax1.set_title(satname, fontweight="bold", fontsize=14)
    ax1.axis("off")

    # add legend to the 2nd plot
    ax2.legend(
        handles=[shoreline_line, buffer_patch, otsu_patch_1, otsu_patch_2],
        loc="lower right",
        fontsize=9,
    )

    # Convert to int for display: False=0, True=1
    im_labels_display = im_labels.astype(int)

    # Use same custom colormap
    from matplotlib.colors import ListedColormap

    otsu_cmap = ListedColormap(["yellow", "blue"])
    ax2.imshow(im_labels_display, cmap=otsu_cmap, alpha=0.3)

    # Second plot: SAR image + Otsu segmentation + overlays
    ax2.imshow(im_display, cmap="gray")
    # ax2.imshow(im_labels, cmap="autumn", alpha=0.3)  # plot the Otsu segmentation
    ax2.imshow(im_labels_display, cmap=otsu_cmap, alpha=0.3)
    if im_ref_buffer is not None:
        ax2.imshow(
            np.ma.masked_where(~im_ref_buffer, im_ref_buffer), cmap="PiYG", alpha=0.5
        )
    ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "r.", markersize=1)
    ax2.set_title(date, fontweight="bold", fontsize=14)
    ax2.axis("off")

    # Save output
    if settings["save_figure"]:
        fig.savefig(
            os.path.join(filepath, f"{date}_{satname}.jpg"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)
    return False


def show_detection(
    im_ms,
    cloud_mask,
    im_labels,
    shoreline,
    image_epsg,
    georef,
    settings,
    date,
    satname,
    im_ref_buffer=None,
    output_directory: str = None,
    shoreline_extraction_area: np.ndarray = None,
    sar_image: bool = False,
):
    """
    Shows the detected shoreline to the user for visual quality control.
    The user can accept/reject the detected shorelines  by using keep/skip
    buttons.

    KV WRL 2018

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'check_detection': bool
            if True, lets user manually accept/reject the mapped shorelines
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline
    im_ref_buffer
        binary image containing a buffer around the reference shoreline
    output_directory: str
        path to the output directory to save the jpg file. If none, the jpg file will be saved at the same location as the input image.
        The jpg files will be saved at output_directory/jpg_files/detection or if output_directory is None, at filepath/jpg_files/detection


    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise

    """

    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    if output_directory is not None:
        # subfolder where the .jpg file is stored if the user accepts the shoreline detection
        filepath = os.path.join(output_directory, "jpg_files", "detection")
        if not os.path.exists(filepath):
            os.makedirs(filepath)
    else:
        # subfolder where the .jpg file is stored if the user accepts the shoreline detection
        filepath = os.path.join(filepath_data, sitename, "jpg_files", "detection")

        if not os.path.exists(filepath):
            os.makedirs(filepath)

    if sar_image:
        im_RGB = im_ms
    else:
        im_RGB = SDS_preprocess.rescale_image_intensity(
            im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9
        )

    # compute classified image
    im_class = np.copy(im_RGB)
    cmap = plt.get_cmap("tab20c")
    colorpalette = cmap(np.arange(0, 13, 1))
    colours = np.zeros((3, 4))
    colours[0, :] = colorpalette[5]
    colours[1, :] = np.array([204 / 255, 1, 1, 1])
    colours[2, :] = np.array([0, 91 / 255, 1, 1])

    # check the shape of im_labels
    if im_labels.ndim == 2:
        # if im_labels is 2D, add a third dimension with only one channel
        im_labels = np.expand_dims(im_labels, axis=2)
        im_class = im_labels.copy()

    else:
        for k in range(0, im_labels.shape[2]):
            im_class[im_labels[:, :, k], 0] = colours[k, 0]
            im_class[im_labels[:, :, k], 1] = colours[k, 1]
            im_class[im_labels[:, :, k], 2] = colours[k, 2]

    # transform world coordinates of shoreline into pixel coordinates
    # use try/except in case there are no coordinates to be transformed (shoreline = [])
    try:
        sl_pix = SDS_tools.convert_world2pix(
            SDS_tools.convert_epsg(shoreline, settings["output_epsg"], image_epsg)[
                :, [0, 1]
            ],
            georef,
        )
    except:
        # if try fails, just add nan into the shoreline vector so the next parts can still run
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])

    shoreline_extraction_area_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    shoreline_extraction_area_pix = []
    if shoreline_extraction_area is not None:
        if len(shoreline_extraction_area) == 0:
            shoreline_extraction_area = None

    if shoreline_extraction_area is not None:
        shoreline_extraction_area_pix = []
        for idx in range(len(shoreline_extraction_area)):
            shoreline_extraction_area_pix.append(
                SDS_preprocess.transform_world_coords_to_pixel_coords(
                    shoreline_extraction_area[idx],
                    settings["output_epsg"],
                    georef,
                    image_epsg,
                )
            )

    if plt.get_fignums():
        # get open figure if it exists
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        # according to the image shape, decide whether it is better to have the images
        # in vertical subplots or horizontal subplots
        if im_RGB.shape[1] > 2.5 * im_RGB.shape[0]:
            # vertical subplots
            gs = gridspec.GridSpec(3, 1)
            gs.update(bottom=0.03, top=0.97, left=0.03, right=0.97)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[2, 0], sharex=ax1, sharey=ax1)
        else:
            # horizontal subplots
            gs = gridspec.GridSpec(1, 3)
            gs.update(bottom=0.05, top=0.95, left=0.05, right=0.95)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
            ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    ax1.imshow(im_RGB)
    ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for idx in range(len(shoreline_extraction_area_pix)):
        ax1.plot(
            shoreline_extraction_area_pix[idx][:, 0],
            shoreline_extraction_area_pix[idx][:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax1.axis("off")
    ax1.set_title(sitename, fontweight="bold", fontsize=16)

    # color map for the reference shoreline buffer
    masked_cmap = plt.get_cmap("PiYG")
    # masked_cmap.set_bad(color="gray", alpha=0)
    # Create a masked array where False values are masked
    masked_array = None
    if im_ref_buffer is not None:
        masked_array = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)

    # create image 2 (classification)
    ax2.imshow(im_class)
    if masked_array is not None:
        ax2.imshow(masked_array, cmap=masked_cmap, alpha=0.60)
    ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for idx in range(len(shoreline_extraction_area_pix)):
        ax2.plot(
            shoreline_extraction_area_pix[idx][:, 0],
            shoreline_extraction_area_pix[idx][:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax2.axis("off")
    orange_patch = mpatches.Patch(color=colours[0, :], label="sand")
    white_patch = mpatches.Patch(color=colours[1, :], label="whitewater")
    blue_patch = mpatches.Patch(color=colours[2, :], label="water")
    black_line = mlines.Line2D([], [], color="k", linestyle="-", label="shoreline")
    buffer_patch = mpatches.Patch(
        color="#800000", alpha=0.80, label="reference shoreline buffer"
    )
    if shoreline_extraction_area is not None:
        shoreline_extraction_area_line = mlines.Line2D(
            [], [], color="#cb42f5", linestyle="-", label="shoreline extraction area"
        )
        ax2.legend(
            handles=[
                orange_patch,
                white_patch,
                blue_patch,
                black_line,
                shoreline_extraction_area_line,
                buffer_patch,
            ],
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
        )
    else:
        ax2.legend(
            handles=[orange_patch, white_patch, blue_patch, black_line, buffer_patch],
            bbox_to_anchor=(1, 0.5),
            fontsize=10,
        )
    ax2.set_title(date, fontweight="bold", fontsize=16)

    if not sar_image:
        # compute MNDWI grayscale image
        im_mwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
        # create image 3 (MNDWI)
        ax3.imshow(im_mwi, cmap="bwr")
        if masked_array is not None:
            ax3.imshow(masked_array, cmap=masked_cmap, alpha=0.60)
        ax3.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
        for idx in range(len(shoreline_extraction_area_pix)):
            ax3.plot(
                shoreline_extraction_area_pix[idx][:, 0],
                shoreline_extraction_area_pix[idx][:, 1],
                color="#cb42f5",
                markersize=1,
            )
        ax3.axis("off")
        ax3.set_title(satname, fontweight="bold", fontsize=16)

    # additional options
    #    ax1.set_anchor('W')
    #    ax2.set_anchor('W')
    #    cb = plt.colorbar()
    #    cb.ax.tick_params(labelsize=10)
    #    cb.set_label('MNDWI values')
    #    ax3.set_anchor('W')

    # if check_detection is True, let user manually accept/reject the images
    skip_image = False
    if settings["check_detection"]:
        # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
        # this variable needs to be immuatable so we can access it after the keypress event
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
                transform=ax1.transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            btn_skip = plt.text(
                -0.1,
                0.9,
                "⇦ skip",
                size=12,
                ha="left",
                va="top",
                transform=ax1.transAxes,
                bbox=dict(boxstyle="square", ec="k", fc="w"),
            )
            btn_esc = plt.text(
                0.5,
                0,
                "<esc> to quit",
                size=12,
                ha="center",
                va="top",
                transform=ax1.transAxes,
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

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings["save_figure"] and not skip_image:
        fig.savefig(
            os.path.join(filepath, date + "_" + satname + ".jpg"),
            dpi=150,
            bbox_inches="tight",
        )

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image


def adjust_detection(
    im_ms,
    cloud_mask,
    im_nodata,
    im_labels,
    im_ref_buffer,
    image_epsg,
    georef,
    settings,
    date,
    satname,
):
    """
    Advanced version of show detection where the user can adjust the detected
    shorelines with a slide bar.

    KV WRL 2020

    Arguments:
    -----------
    im_ms: np.array
        RGB + downsampled NIR and SWIR
    cloud_mask: np.array
        2D cloud mask with True where cloud pixels are
    im_labels: np.array
        3D image containing a boolean image for each class in the order (sand, swash, water)
    im_ref_buffer: np.array
        Binary image containing a buffer around the reference shoreline
    image_epsg: int
        spatial reference system of the image from which the contours were extracted
    georef: np.array
        vector of 6 elements [Xtr, Xscale, Xshear, Ytr, Yshear, Yscale]
    date: string
        date at which the image was taken
    satname: string
        indicates the satname (L5,L7,L8 or S2)
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)
        'output_epsg': int
            output spatial reference system as EPSG code
        'save_figure': bool
            if True, saves a -jpg file for each mapped shoreline

    Returns:
    -----------
    skip_image: boolean
        True if the user wants to skip the image, False otherwise
    shoreline: np.array
        array of points with the X and Y coordinates of the shoreline
    t_mndwi: float
        value of the MNDWI threshold used to map the shoreline

    """

    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    # subfolder where the .jpg file is stored if the user accepts the shoreline detection
    filepath = os.path.join(filepath_data, sitename, "jpg_files", "detection")
    # format date
    date_str = datetime.strptime(date, "%Y-%m-%d-%H-%M-%S").strftime(
        "%Y-%m-%d  %H:%M:%S"
    )
    im_RGB = SDS_preprocess.rescale_image_intensity(
        im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9
    )

    # compute classified image
    im_class = np.copy(im_RGB)
    cmap = plt.get_cmap("tab20c")
    colorpalette = cmap(np.arange(0, 13, 1))
    colours = np.zeros((3, 4))
    colours[0, :] = colorpalette[5]
    colours[1, :] = np.array([204 / 255, 1, 1, 1])
    colours[2, :] = np.array([0, 91 / 255, 1, 1])
    for k in range(0, im_labels.shape[2]):
        im_class[im_labels[:, :, k], 0] = colours[k, 0]
        im_class[im_labels[:, :, k], 1] = colours[k, 1]
        im_class[im_labels[:, :, k], 2] = colours[k, 2]

    # compute MNDWI grayscale image
    im_mndwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    # buffer MNDWI using reference shoreline
    im_mndwi_buffer = np.copy(im_mndwi)
    im_mndwi_buffer[~im_ref_buffer] = np.nan

    # get MNDWI pixel intensity in each class (for histogram plot)
    int_sand = im_mndwi[im_labels[:, :, 0]]
    int_ww = im_mndwi[im_labels[:, :, 1]]
    int_water = im_mndwi[im_labels[:, :, 2]]
    labels_other = np.logical_and(
        np.logical_and(~im_labels[:, :, 0], ~im_labels[:, :, 1]), ~im_labels[:, :, 2]
    )
    int_other = im_mndwi[labels_other]

    # create figure
    if plt.get_fignums():
        # if it exists, open the figure
        fig = plt.gcf()
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax4 = fig.axes[3]
    else:
        # else create a new figure
        fig = plt.figure()
        fig.set_size_inches([18, 9])
        mng = plt.get_current_fig_manager()
        gs = gridspec.GridSpec(2, 3, height_ratios=[4, 1])
        gs.update(bottom=0.05, top=0.95, left=0.03, right=0.97)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
        ax3 = fig.add_subplot(gs[0, 2], sharex=ax1, sharey=ax1)
        ax4 = fig.add_subplot(gs[1, :])
    ##########################################################################
    # to do: rotate image if too wide
    ##########################################################################

    # change the color of nans to either black (0.0) or white (1.0) or somewhere in between
    nan_color = 1.0
    im_RGB = np.where(np.isnan(im_RGB), nan_color, im_RGB)
    im_class = np.where(np.isnan(im_class), 1.0, im_class)

    # plot image 1 (RGB)
    ax1.imshow(im_RGB)
    ax1.axis("off")
    ax1.set_title("%s - %s" % (sitename, satname), fontsize=12)

    # plot image 2 (classification)
    ax2.imshow(im_class)
    ax2.axis("off")
    orange_patch = mpatches.Patch(color=colours[0, :], label="sand")
    white_patch = mpatches.Patch(color=colours[1, :], label="whitewater")
    blue_patch = mpatches.Patch(color=colours[2, :], label="water")
    black_line = mlines.Line2D([], [], color="k", linestyle="-", label="shoreline")
    ax2.legend(
        handles=[orange_patch, white_patch, blue_patch, black_line],
        bbox_to_anchor=(1.1, 0.5),
        fontsize=10,
    )
    ax2.set_title(date_str, fontsize=12)

    # plot image 3 (MNDWI)
    ax3.imshow(im_mndwi, cmap="bwr")
    ax3.axis("off")
    ax3.set_title("MNDWI", fontsize=12)

    # plot histogram of MNDWI values
    binwidth = 0.01
    ax4.set_facecolor("0.75")
    ax4.yaxis.grid(color="w", linestyle="--", linewidth=0.5)
    ax4.set(ylabel="PDF", yticklabels=[], xlim=[-1, 1])
    if len(int_sand) > 0 and sum(~np.isnan(int_sand)) > 0:
        bins = np.arange(np.nanmin(int_sand), np.nanmax(int_sand) + binwidth, binwidth)
        ax4.hist(int_sand, bins=bins, density=True, color=colours[0, :], label="sand")
    if len(int_ww) > 0 and sum(~np.isnan(int_ww)) > 0:
        bins = np.arange(np.nanmin(int_ww), np.nanmax(int_ww) + binwidth, binwidth)
        ax4.hist(
            int_ww,
            bins=bins,
            density=True,
            color=colours[1, :],
            label="whitewater",
            alpha=0.75,
        )
    if len(int_water) > 0 and sum(~np.isnan(int_water)) > 0:
        bins = np.arange(
            np.nanmin(int_water), np.nanmax(int_water) + binwidth, binwidth
        )
        ax4.hist(
            int_water,
            bins=bins,
            density=True,
            color=colours[2, :],
            label="water",
            alpha=0.75,
        )
    if len(int_other) > 0 and sum(~np.isnan(int_other)) > 0:
        bins = np.arange(
            np.nanmin(int_other), np.nanmax(int_other) + binwidth, binwidth
        )
        ax4.hist(
            int_other, bins=bins, density=True, color="C4", label="other", alpha=0.5
        )

    # automatically map the shoreline based on the classifier if enough sand pixels
    try:
        if sum(sum(im_labels[:, :, 0])) > 50:
            # use classification to refine threshold and extract the sand/water interface
            contours_mndwi, t_mndwi = find_wl_contours2(
                im_ms, im_labels, cloud_mask, im_ref_buffer
            )
        else:
            # find water contours on MNDWI grayscale image
            contours_mndwi, t_mndwi = find_wl_contours1(
                im_mndwi, cloud_mask, im_ref_buffer
            )
    except:
        print("Could not map shoreline so image was skipped")
        # clear axes and return skip_image=True, so that image is skipped above
        for ax in fig.axes:
            ax.clear()
        return True, [], []

    # process the water contours into a shoreline
    cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)
    shoreline = process_shoreline(
        contours_mndwi,
        cloud_mask_adv,
        im_nodata,
        georef,
        image_epsg,
        settings,
    )
    # convert shoreline to pixels
    if len(shoreline) > 0:
        sl_pix = SDS_tools.convert_world2pix(
            SDS_tools.convert_epsg(shoreline, settings["output_epsg"], image_epsg)[
                :, [0, 1]
            ],
            georef,
        )
    else:
        sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    # plot the shoreline on the images
    sl_plot1 = ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=3)
    sl_plot2 = ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=3)
    sl_plot3 = ax3.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=3)
    t_line = ax4.axvline(x=t_mndwi, ls="--", c="k", lw=1.5, label="threshold")
    ax4.legend(loc=1)
    plt.draw()  # to update the plot
    # adjust the threshold manually by letting the user change the threshold
    ax4.set_title(
        "Click on the plot below to change the location of the threhsold and adjust the shoreline detection. When finished, press <Enter>"
    )
    while True:
        # let the user click on the threshold plot
        pt = ginput(n=1, show_clicks=True, timeout=-1)
        # if a point was clicked
        if len(pt) > 0:
            # if user clicked somewhere wrong and value is not between -1 and 1
            if np.abs(pt[0][0]) >= 1:
                continue
            # update the threshold value
            t_mndwi = pt[0][0]
            # update the plot
            t_line.set_xdata([t_mndwi, t_mndwi])
            # map contours with new threshold
            contours = measure.find_contours(im_mndwi_buffer, t_mndwi)
            # remove contours that contain NaNs (due to cloud pixels in the contour)
            contours = process_contours(contours)
            # process the water contours into a shoreline
            shoreline = process_shoreline(
                contours, cloud_mask, georef, image_epsg, settings
            )
            # convert shoreline to pixels
            if len(shoreline) > 0:
                sl_pix = SDS_tools.convert_world2pix(
                    SDS_tools.convert_epsg(
                        shoreline, settings["output_epsg"], image_epsg
                    )[:, [0, 1]],
                    georef,
                )
            else:
                sl_pix = np.array([[np.nan, np.nan], [np.nan, np.nan]])
            # update the plotted shorelines
            sl_plot1[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot2[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            sl_plot3[0].set_data([sl_pix[:, 0], sl_pix[:, 1]])
            fig.canvas.draw_idle()
        else:
            ax4.set_title("MNDWI pixel intensities and threshold")
            break

    # let user manually accept/reject the image
    skip_image = False
    # set a key event to accept/reject the detections (see https://stackoverflow.com/a/15033071)
    # this variable needs to be immuatable so we can access it after the keypress event
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
            transform=ax1.transAxes,
            bbox=dict(boxstyle="square", ec="k", fc="w"),
        )
        btn_skip = plt.text(
            -0.1,
            0.9,
            "⇦ skip",
            size=12,
            ha="left",
            va="top",
            transform=ax1.transAxes,
            bbox=dict(boxstyle="square", ec="k", fc="w"),
        )
        btn_esc = plt.text(
            0.5,
            0,
            "<esc> to quit",
            size=12,
            ha="center",
            va="top",
            transform=ax1.transAxes,
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

    # if save_figure is True, save a .jpg under /jpg_files/detection
    if settings["save_figure"] and not skip_image:
        fig.savefig(os.path.join(filepath, date + "_" + satname + ".jpg"), dpi=150)

    # don't close the figure window, but remove all axes and settings, ready for next plot
    for ax in fig.axes:
        ax.clear()

    return skip_image, shoreline, t_mndwi
