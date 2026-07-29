"""
This module contains all the functions needed to download the satellite images
from the Google Earth Engine server

Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# Standard library imports
import ast
import hashlib
import json
import logging
import os
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple, Union, Optional
import zipfile
import functools

# Third-party imports
import ee
import google.auth
import matplotlib.pyplot as plt
import numpy as np
import pytz
import requests
from tqdm.auto import tqdm

# raise an error in case gdal is missing
try:
    from osgeo import gdal
except ModuleNotFoundError as missing_gdal:
    print(
        "GDAL is not installed. Please install GDAL by running 'conda install -c conda-forge gdal -y' "
    )
    raise missing_gdal

# CoastSat modules
from coastsat import SDS_preprocess, SDS_tools

np.seterr(all="ignore")  # raise/ignore divisions by 0 and nans
gdal.PushErrorHandler("CPLQuietErrorHandler")

# Sentinel-1 acquisition modes and orbit directions accepted in sentinel_1_properties.
VALID_S1_INSTRUMENT_MODES = ("IW", "EW", "SM", "WV")
VALID_S1_ORBIT_PASSES = ("ASCENDING", "DESCENDING")


def get_skip_cache_path(inputs: dict) -> str:
    """
    Construct the file path to the skip cache JSON file for a given site.

    Args:
        inputs (dict): A dictionary containing configuration parameters with keys:
            - 'filepath' (str): The base file path where site directories are stored.
            - 'sitename' (str): The name of the site.

    Returns:
        str: The full path to the skip_cache.json file for the specified site.

    Example:
        >>> inputs = {'filepath': '/data/sites', 'sitename': 'beach_01'}
        >>> get_skip_cache_path(inputs)
        '/data/sites/beach_01/skip_cache.json'
    """
    site_dir = os.path.join(inputs["filepath"], inputs["sitename"])
    return os.path.join(site_dir, "skip_cache.json")


def get_skip_query_signature(inputs: dict) -> str:
    """Build a stable hash for the query context used to scope skip-cache entries."""
    payload = {
        "polygon": inputs.get("polygon", []),
        "dates": inputs.get("dates", []),
        "sat_list": sorted(inputs.get("sat_list", [])),
    }
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def load_skip_cache(inputs: dict) -> dict:
    """
    Load the skip cache from disk or return a default cache structure.
    Attempts to load a cached dictionary containing skip entries
    and settings for a given site. If the cache file doesn't exist or is invalid,
    it returns a default cache structure. The cache is validated to ensure all
    required keys are present and have the correct types.
    Args:
        inputs (dict): A dictionary containing configuration inputs, including
                      the 'sitename' key used to locate the cache file.
    Returns:
        dict: A dictionary with the following structure:
              {
                  "version": int,           # Cache format version
                  "sitename": str,          # Name of the site
                  "settings": dict,         # Site-specific settings
                  "entries": dict           # Cached entries/data
              Returns the loaded and validated cache if available, or a default
              cache structure if not found or invalid.
    Raises:
        None: All exceptions are caught and result in returning the default cache.
    """
    default_cache = {
        "version": 1,
        "sitename": inputs.get("sitename", ""),
        "settings": {},
        "entries": {},
    }
    cache_path = get_skip_cache_path(inputs)
    if not os.path.exists(cache_path):
        return default_cache

    try:
        with open(cache_path, "r", encoding="utf-8") as fp:
            cache = json.load(fp)
        if not isinstance(cache, dict):
            return default_cache
        if "entries" not in cache or not isinstance(cache["entries"], dict):
            cache["entries"] = {}
        if "version" not in cache:
            cache["version"] = 1
        if "sitename" not in cache:
            cache["sitename"] = inputs.get("sitename", "")
        if "settings" not in cache or not isinstance(cache["settings"], dict):
            cache["settings"] = {}
        return cache
    except Exception:
        return default_cache


def save_skip_cache(inputs: dict, cache: dict) -> None:
    """Persist skip cache to disk."""
    cache_path = get_skip_cache_path(inputs)
    SDS_preprocess.write_to_json(cache_path, cache)


def update_skip_cache_metadata(
    skip_cache: dict,
    inputs: dict,
    max_cloud_no_data_cover: float = None,
    max_cloud_cover: float = None,
    s2cloudless_prob: int = 60,
    cloud_mask_issue: bool = False,
) -> None:
    """Update top-level cache metadata describing the active run settings."""
    skip_cache["version"] = 1
    skip_cache["sitename"] = inputs.get("sitename", "")
    skip_cache["settings"] = {
        "max_cloud_no_data_cover(percent_no_data)": max_cloud_no_data_cover,
        "max_cloud_cover": max_cloud_cover,
        "s2cloudless_prob": s2cloudless_prob,
        "cloud_mask_issue": cloud_mask_issue,
    }
    skip_cache["updated_at_utc"] = datetime.now(timezone.utc).isoformat()


def build_skip_cache_key(im_meta: dict, satname: str) -> str:
    """Build a stable key for identifying an EE image across runs."""
    props = im_meta.get("properties", {})
    image_id = im_meta.get("id", "")
    if image_id:
        raw_key = image_id
    else:
        system_index = props.get("system:index", "")
        system_time = props.get("system:time_start", "")
        polarization = props.get("transmitterReceiverPolarisation", "")
        if isinstance(polarization, list):
            polarization = "-".join(polarization)
        raw_key = f"{satname}|{system_index}|{system_time}|{polarization}"

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def get_image_datetime_string(im_meta: dict) -> str:
    """Return acquisition datetime formatted as YYYY-MM-DD-HH-MM-SS."""
    timestamp_ms = im_meta.get("properties", {}).get("system:time_start")
    if timestamp_ms is None:
        return ""
    try:
        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=pytz.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return ""
    return dt.strftime("%Y-%m-%d-%H-%M-%S")


def should_skip_from_cache(
    cache_entry: dict,
    query_signature: str,
    max_cloud_no_data_cover: float = None,
    max_cloud_cover: float = None,
) -> bool:
    """Return True when cached cloud/no-data metrics still fail current thresholds."""
    if not cache_entry or cache_entry.get("skip_type") != "cloud_nodata":
        return False

    if cache_entry.get("query_signature") != query_signature:
        return False

    cloud_cover_combined = cache_entry.get("cloud_cover_combined")
    cloud_cover = cache_entry.get("cloud_cover")
    if cloud_cover_combined is None or cloud_cover is None:
        return False

    if (
        max_cloud_no_data_cover is not None
        and cloud_cover_combined > max_cloud_no_data_cover
    ):
        return True

    if max_cloud_cover is not None and cloud_cover > max_cloud_cover:
        return True

    return False


def prune_skip_cache_for_current_thresholds(
    skip_cache: dict,
    query_signature: str,
    max_cloud_no_data_cover: float = None,
    max_cloud_cover: float = None,
) -> int:
    """Remove stale skip-cache entries that no longer fail active thresholds."""
    entries = skip_cache.get("entries", {})
    keys_to_remove = []

    for cache_key, cache_entry in entries.items():
        if cache_entry.get("query_signature") != query_signature:
            continue
        if cache_entry.get("skip_type") != "cloud_nodata":
            continue

        if not should_skip_from_cache(
            cache_entry,
            query_signature,
            max_cloud_no_data_cover=max_cloud_no_data_cover,
            max_cloud_cover=max_cloud_cover,
        ):
            keys_to_remove.append(cache_key)

    for cache_key in keys_to_remove:
        entries.pop(cache_key, None)

    return len(keys_to_remove)


def record_skip_cache_entry(
    skip_cache: dict,
    cache_key: str,
    im_meta: dict,
    satname: str,
    query_signature: str,
    metrics: dict,
) -> None:
    """Record a cloud/no-data skip decision and measured values in cache."""
    skip_cache.setdefault("entries", {})
    skip_cache["entries"][cache_key] = {
        "image_id": im_meta.get("id", ""),
        "image_datetime": get_image_datetime_string(im_meta),
        "satname": satname,
        "skip_type": "cloud_nodata",
        "query_signature": query_signature,
        "cloud_cover_combined": float(metrics.get("cloud_cover_combined", 0.0)),
        "cloud_cover": float(metrics.get("cloud_cover", 0.0)),
        "reason": metrics.get("reason", "cloud_nodata"),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def get_metadata_combined(inputs: Dict[str, Any]) -> Dict[str, Dict[str, list]]:
    """
    Extracts and compiles metadata from multiple satellite directories within a given site folder,
    filtering by date range and expected metadata structure. The compiled metadata is then saved
    to a JSON file within the site directory.

    Parameters:
    ----------
    inputs : dict
        A dictionary containing the following keys:
            - "filepath" (str): Base directory path where the site folder is located.
            - "sitename" (str): Name of the site directory.
            - "dates" (tuple): A tuple of start and end dates (datetime or string) to filter metadata files.
            - "sat_list" (list, optional): List of satellite names to process. Defaults to ["L5", "L7", "L8", "L9", "S1", "S2"].

    Returns:
    -------
    Dict[str, Dict[str, list]]
        A nested dictionary where the first-level keys are satellite names and the second-level keys are metadata fields,
        each mapping to a list of values collected from metadata files.

    Raises:
    ------
    FileNotFoundError
        If the specified site directory does not exist.

    Notes:
    -----
    - Assumes the presence of a `meta` folder within each satellite directory under the site folder.
    - Each metadata file is expected to contain a "filename" field to parse the acquisition date.
    - Filters out metadata entries that fall outside the specified date range.
    - Saves the final compiled metadata as a JSON file named `{sitename}_metadata.json` in the site directory.

    """
    filepath = os.path.join(inputs["filepath"], inputs["sitename"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The directory {filepath} does not exist.")

    metadata = dict()
    satellite_list = inputs.get("sat_list", ["L5", "L7", "L8", "L9", "S1", "S2"])

    for satname in satellite_list:
        sat_path = os.path.join(filepath, satname)
        if not os.path.exists(sat_path):
            continue

        filepath_meta = os.path.join(sat_path, "meta")
        if not os.path.exists(filepath_meta):
            continue

        metadata[satname] = {}
        filenames_meta = sorted(os.listdir(filepath_meta))
        for im_meta in filenames_meta:
            full_path = os.path.join(filepath_meta, im_meta)
            meta = read_metadata_file(full_path)

            # Check if filename is valid and within date range
            date = parse_date_from_filename(meta.get("filename", ""))
            if not date:
                continue
            start_date = format_date(inputs["dates"][0])
            end_date = format_date(inputs["dates"][1])
            if not (start_date <= date <= end_date):
                continue

            # Initialize keys dynamically if not already
            for key, val in meta.items():
                if key not in metadata[satname]:
                    metadata[satname][key] = []
                metadata[satname][key].append(val)

            # Always store parsed date
            if "dates" not in metadata[satname]:
                metadata[satname]["dates"] = []
            metadata[satname]["dates"].append(date)

            # Calculate im_dimensions if both height and width are present
            if "im_height" in meta and "im_width" in meta:
                if "im_dimensions" not in metadata[satname]:
                    metadata[satname]["im_dimensions"] = []
                metadata[satname]["im_dimensions"].append(
                    [meta["im_height"], meta["im_width"]]
                )
            elif "im_dimensions" not in metadata[satname]:
                metadata[satname]["im_dimensions"] = []

    # Save metadata
    metadata_path = os.path.join(filepath, f"{inputs['sitename']}_metadata.json")
    SDS_preprocess.write_to_json(metadata_path, metadata)
    return metadata


def process_image_metadata(
    im_meta: dict,
) -> tuple[str, str]:
    """
    Extracts image date, EPSG code, and orbit pass direction from metadata.

    Args:
        im_meta (dict): Metadata dictionary of the image.
        This is the metadata dictionary returned by the GEE API.
        It generally contains these 3 main keys that contain subdata we are interested in:
        - properties: contains the time_start and orbitProperties_pass
        - bands: contains the crs (EPSG code) of the image
            - In this case we are interested in the first band (0) of the image which should contain the VH band
        - id : contains the image ID

    Returns:
        tuple[str, str]: Formatted date string, and orbit direction.
    """
    t = im_meta["properties"]["system:time_start"]
    im_timestamp = datetime.fromtimestamp(t / 1000, tz=pytz.utc)
    im_date = im_timestamp.strftime("%Y-%m-%d-%H-%M-%S")
    return im_date


def get_file_name(
    im_date: str,
    satname: str,
    sitename: str,
    polar: str,
    suffix: str,
    all_names: list[str],
) -> str:
    """
    Generates a unique filename and handles duplicates.

    Args:
        im_date (str): Formatted image date.
        satname (str): Satellite name.
        sitename (str): Site name.
        polar (str): Polarization.
        suffix (str): File suffix.
        all_names (list[str]): List of already used filenames.

    Returns:
        str: Unique filename.
    """
    base_name = f"{im_date}_{satname}_{sitename}_{polar}{suffix}"
    filename = base_name
    duplicate_counter = 0
    while filename in all_names:
        filename = (
            f"{im_date}_{satname}_{sitename}_{polar}_dup{duplicate_counter}{suffix}"
        )
        duplicate_counter += 1
    all_names.append(filename)
    return filename


def write_metadata_file(
    filepath: str, filename_txt: str, metadata: dict, polar: str = None
) -> None:
    """
    Writes metadata to a text file.

    Args:
        filepath (str): Path to save the metadata file.
        filename_txt (str): Base name of the file.
        metadata (dict): Metadata dictionary.
        polar (str, optional): legacy per-polarization suffix. Leave unset so the
            file is named like the optical ones ('{date}_{sat}_{site}.txt'); a
            Sentinel-1 scene gets one metadata file covering all its polarizations.
    """
    name = f"{filename_txt}_{polar}.txt" if polar else f"{filename_txt}.txt"
    with open(os.path.join(filepath, name), "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}\t{value}\n")


def write_image_metadata(
    fp_ms,
    im_fn,
    im_meta,
    filename_ms,
    im_epsg,
    accuracy_georef,
    image_quality,
    output_dir,
    logger,
):
    """
    Extracts image dimensions and writes image metadata to a text file.
    This creates the metadata txt file for the multispectral image. This does not work for radar like S1 images.

    Parameters:
        fp_ms (str): Path to the multispectral images folder.
        im_fn (dict): Dictionary with image filename keys, including 'ms'.
        im_meta (dict): Metadata dictionary containing at least the image ID.
        filename_ms (str): Full multispectral image filename.
        im_epsg (int): EPSG code for the image projection.
        accuracy_georef (str): Accuracy of the georeferencing.
        image_quality (str): Quality of the image.
        output_dir (str): Directory to save the metadata file.
            Default input is Filepaths[0] this is the meta folder location
        logger (Logger): Logger instance to record success messages.

    Raises:
        Exception: If 'ms' filename is missing or unknown in `im_fn`.
    """

    ms_filename = im_fn.get("ms")
    image_id = im_meta.get("id", "unknown")

    if not fp_ms or not ms_filename or ms_filename == "unknown":
        raise Exception(f"Could not find ms band filename for image ID: {image_id}")

    image_path = os.path.join(fp_ms, ms_filename)  # path to multispectral tiff
    width, height = SDS_tools.get_image_dimensions(image_path)

    metadata = {
        "filename": filename_ms,
        "epsg": im_epsg,
        "acc_georef": accuracy_georef,
        "image_quality": image_quality,
        "im_width": width,
        "im_height": height,
    }

    output_filename = ms_filename.replace("_ms", "").replace(".tif", ".txt")
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}\t{value}\n")

    logger.info(f"Successfully downloaded image ID {image_id} as {ms_filename}")


def authenticate_and_initialize(project=""):
    """
    Authenticates and initializes the Earth Engine API.
    This function handles the authentication and initialization process:
        1. Try to use existing token to initialize
        2. If 1 fails, try to refresh the token using Application Default Credentials
        3. If 2 fails, authenticate manually via the web browser
    args:
        project (str): The Google Cloud project ID to use for Earth Engine.
    Raises:
        Exception: If the project ID is not provided or if authentication fails.
    Returns:
        None
    """
    if project == "":
        raise Exception("Please provide a Google project ID that has access to GEE.")
    # first try to initialize connection with GEE server with existing token
    try:
        initialize_gee(project=project)
    except:
        print(
            "Google Earth Engine is not initialized. Attempting to authenticate and initialize with Google Earth Engine API"
        )
        # get the user to authenticate manually and initialize the sesion
        ee.Authenticate()
        ee.Initialize(project=project)
        print("GEE initialized (manual authentication).")


# added from coastsat on 10/24/2024 modified by Sharon to make this function compatible with coastseg
def initialize_gee(project=""):
    """
    Initializes the Earth Engine API.
    This function handles the initialization process:
        1. Try to use existing token to initialize
        2. If 1 fails, try to refresh the token using Application Default Credentials
    args:
        project (str): The Google Cloud project ID to use for Earth Engine.
    Raises:
        Exception: If the project ID is not provided or if authentication fails.
    Returns:
        None
    """
    if project == "":
        raise Exception("Please provide a Google project ID that has access to GEE.")
    # first try to initialize connection with GEE server with existing token
    try:
        print(
            f"Google Earth Engine(GEE) is not initialized. Attempting to initialize for project ID '{project}'"
        )
        ee.Initialize(project=project)
        print("GEE initialized with existing token.")
    except:
        # if token is expired, try to refresh it using the existing credentials
        try:
            print(
                "Google Earth Engine is not initialized. Attempting to initialize with Google Earth Engine API with existing credentials"
            )
            creds = ee.data.get_persistent_credentials()
            auth_req = google.auth.transport.requests.Request()
            creds.refresh(auth_req)
            # initialise GEE session with refreshed credentials
            ee.Initialize(creds, project=project)
            print("GEE initialized (refreshed token).")
        except:
            print(
                f"Unable to initialize Google Earth Engine connection. Please authenticate with Google Earth Engine."
            )
            raise Exception(
                "Unable to initialize Google Earth Engine connection. Please authenticate with Google Earth Engine."
            )


def release_logger(logger):
    """
    Release the logger and its associated file handlers.

    :param logger: The logger object to be released.
    """
    # Remove all handlers associated with the logger
    for handler in logger.handlers[:]:
        # Close the handler if it's a FileHandler
        if isinstance(handler, logging.FileHandler):
            handler.close()
        # Remove the handler from the logger
        logger.removeHandler(handler)


def setup_logger(
    folder,
    base_filename="download_report",
    log_format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
):
    # Determine the next available log file number
    i = 1
    while True:
        log_filename = f"{base_filename}{i}.txt" if i > 1 else f"{base_filename}.txt"
        log_filepath = os.path.join(folder, log_filename)
        if not os.path.exists(log_filepath):
            break
        i += 1

    # Create a custom logger
    logger = logging.getLogger("satellite_download_logger")
    logger.setLevel(logging.INFO)  # Log all levels of messages

    # Create handlers
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    log_format = logging.Formatter(log_format)
    file_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(file_handler)

    return logger


# Custom exception classes
class RequestSizeExceededError(Exception):
    pass


class TooManyRequests(Exception):
    pass


def retry(func):
    @functools.wraps(func)
    def wrapper_retry(*args, **kwargs):
        logger = kwargs.get("logger", None)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # first attempt does not wait
                if attempt > 0:
                    time.sleep(1)
                return func(*args, **kwargs)
            except Exception as e:
                print(
                    f"Attempt {attempt+1}/{max_attempts} failed with error: {type(e).__name__}"
                )
                if logger:
                    logger.warning(
                        f"Retry {attempt + 1}/{max_attempts} for function {func.__name__} with image_id {kwargs.get('image_id', 'N/A')} due to {e}"
                    )
                if attempt == max_attempts - 1:
                    print(
                        f"Max retries {attempt+1}/{max_attempts}  exceeded for {func.__name__} due to {type(e).__name__}"
                    )
                    raise

    return wrapper_retry


def debug_kill_wifi():
    import os

    os.system("netsh wlan disconnect")


def _normalize_dates(dates: list) -> tuple[str, str]:
    """Ensure dates are in string format."""
    start, end = dates
    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d")
    return start, end


def get_images_list_from_collection(
    ee_col: ee.ImageCollection,
    polygon: list,
    dates: list,
    min_roi_coverage: float = 0.30,
    **kwargs,
) -> list:
    """
    Retrieves a list of images from a given Earth Engine collection within a specified polygon and date range.
    Splits the collection if it exceeds the maximum size of 5000 images.

    Args:
        ee_col (ee.ImageCollection): The Earth Engine collection to retrieve images from.
        polygon (list): The coordinates of the polygon as a list of [longitude, latitude] pairs.
        dates (list): The start and end dates of the date range as a list of strings in the format "YYYY-MM-DD".
        min_roi_coverage (float): Minimum required fraction of the polygon area that must be covered by an image.
        **kwargs: Additional keyword arguments to pass to the filtering function.
            min_roi_coverage (float): Minimum required fraction of the polygon area that must be covered by an image.
            tol (float): The error tolerance (in meters) for area and intersection calculations.
        Raises:
            ValueError: If the number of splits is greater than the range of days available.

    Returns:
        list: A list of images in the collection that satisfy the given polygon and date range.
    """
    col = ee_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(
        dates[0], dates[1]
    )
    col = filter_collection_by_coverage(
        col, ee.Geometry.Polygon(polygon), min_roi_coverage=min_roi_coverage, **kwargs
    )

    col_size = col.size().getInfo()

    if col_size <= 4999:
        return col.getInfo().get("features")  # this is im_list

    im_list = []
    # the max size of the collection is 5000, so we need to split the collection if it is larger
    if col_size > 4999:
        num_splits = 2
        split_ranges = split_date_range(dates[0], dates[1], num_splits)
        while True:
            sub_col_sizes = []
            for start_date, end_date in split_ranges:
                sub_col = col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(
                    start_date, end_date
                )
                sub_col_size = sub_col.size().getInfo()
                sub_col_sizes.append(sub_col_size)

            if all(size <= 4999 for size in sub_col_sizes):
                break
            num_splits += 1
            split_ranges = split_date_range(dates[0], dates[1], num_splits)

        for start_date, end_date in split_ranges:
            sub_col = ee_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(
                start_date, end_date
            )
            im_list.extend(sub_col.getInfo().get("features"))

    return im_list


def split_date_range(start_date, end_date, num_splits):
    """
    Splits a date range into multiple smaller date ranges.

    Args:
        start_date (str): The start date of the range in the format 'YYYY-MM-DD'.
        end_date (str): The end date of the range in the format 'YYYY-MM-DD'.
        num_splits (int): The number of splits to create.

    Returns:
        list: A list of tuples representing the split date ranges. Each tuple contains the start date and end date of a split range in the format 'YYYY-MM-DD'.

    Raises:
        ValueError: If the number of splits is greater than the range of days available.
    """
    # Check if start_date and end_date are strings and convert them if necessary
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate the total duration of the range
    total_duration = (end_date - start_date).days

    if total_duration < num_splits - 1:
        raise ValueError(
            "The number of splits is greater than the range of days available."
        )

    # Calculate the duration of each split
    split_duration = total_duration // num_splits

    # Generate the split date ranges
    split_ranges = []
    for i in range(num_splits):
        split_start_date = start_date + timedelta(days=i * split_duration)
        if i == num_splits - 1:
            # Ensure the last range ends exactly at the end_date
            split_end_date = end_date
        else:
            split_end_date = split_start_date + timedelta(days=split_duration)

        # Append the split range as a tuple
        split_ranges.append(
            (split_start_date.strftime("%Y-%m-%d"), split_end_date.strftime("%Y-%m-%d"))
        )

    return split_ranges


DEFAULT_SENTINEL_1_PROPERTIES = {
    "transmitterReceiverPolarisation": list(SDS_tools.DEFAULT_SAR_POLARIZATIONS),
    "instrumentMode": "IW",
}


def get_sentinel1_polarizations(inputs, default_properties=None, logger=None) -> list:
    """
    Returns the Sentinel-1 polarizations to download, in canonical order.

    Defaults to SDS_tools.DEFAULT_SAR_POLARIZATIONS (VV and VH), which is what the
    [VV, VH, VV-VH] composite needs. The requested list is normalised to
    SDS_tools.SAR_POLARIZATIONS order so the folder layout and the channel order of
    im_ms never depend on the order the user happened to list them in.

    Args:
        inputs (dict): may contain 'sentinel_1_properties'
        default_properties (dict, optional): defaults to DEFAULT_SENTINEL_1_PROPERTIES
        logger (optional): used to warn when no properties were provided

    Returns:
        list of str: polarizations in SAR_POLARIZATIONS order, e.g. ['VV', 'VH']

    Raises:
        ValueError: if the list is empty or contains an unknown polarization
    """
    defaults = default_properties or DEFAULT_SENTINEL_1_PROPERTIES
    user_properties = inputs.get("sentinel_1_properties") or {}
    if not user_properties and logger is not None:
        logger.warning(
            "No user-defined Sentinel-1 properties provided. Using default values."
        )
    # user values override defaults key by key
    merged_properties = {**defaults, **user_properties}

    polarizations = merged_properties.get(
        "transmitterReceiverPolarisation", SDS_tools.DEFAULT_SAR_POLARIZATIONS
    )
    if isinstance(polarizations, str):
        polarizations = [polarizations]
    requested = {
        str(polar).strip().upper() for polar in polarizations if str(polar).strip()
    }
    if not requested:
        raise ValueError(
            "sentinel_1_properties['transmitterReceiverPolarisation'] must list "
            "at least one polarization."
        )

    invalid_pols = sorted(requested - set(SDS_tools.SAR_POLARIZATIONS))
    if invalid_pols:
        raise ValueError(
            f"Invalid polarizations: {invalid_pols}. "
            f"Valid options are: {list(SDS_tools.SAR_POLARIZATIONS)}"
        )

    return [polar for polar in SDS_tools.SAR_POLARIZATIONS if polar in requested]


def create_polarization_filter(polarizations, mode: str = "and"):
    """
    Creates an Earth Engine filter for the specified polarizations.

    Args:
        polarizations: str or list of str
            Single polarization (e.g., 'VV') or list of polarizations (e.g., ['VV', 'VH'])
        mode: str
            'and' (default) keeps only scenes carrying EVERY requested polarization,
            which is what downloading a matched set of polarizations requires.
            'or' keeps scenes carrying any of them.

    Returns:
        ee.Filter: Combined filter for the specified polarizations
    """
    # Convert single polarization to list if needed
    if isinstance(polarizations, str):
        polarizations = [polarizations]
    polarizations = list(polarizations)
    if not polarizations:
        raise ValueError("At least one polarization must be requested.")

    # Validate polarizations (kept a list so the message is deterministically ordered)
    valid_pols = SDS_tools.SAR_POLARIZATIONS
    invalid_pols = [p for p in polarizations if p not in valid_pols]
    if invalid_pols:
        raise ValueError(
            f"Invalid polarizations: {invalid_pols}. "
            f"Valid options are: {list(valid_pols)}"
        )

    # Create individual filters
    polar_filters = [
        ee.Filter.listContains("transmitterReceiverPolarisation", p)
        for p in polarizations
    ]
    if len(polar_filters) == 1:
        return polar_filters[0]

    mode = str(mode).lower()
    if mode == "and":
        return ee.Filter.And(*polar_filters)
    if mode == "or":
        return ee.Filter.Or(*polar_filters)
    raise ValueError(
        f"Unknown polarization filter mode: {mode!r} (expected 'and' or 'or')."
    )


def create_sentinel1_metadata_filters(properties):
    """
    Builds the ee.Filters for the Sentinel-1 scene metadata in sentinel_1_properties.

    Covers the two properties other than the polarizations (which have their own
    filter, see create_polarization_filter): 'instrumentMode' and
    'orbitProperties_pass'. A property that is absent or empty contributes no filter,
    so orbit direction stays unrestricted unless the user asks for one.

    Args:
        properties (dict): merged sentinel_1_properties, e.g.
            {"instrumentMode": "IW", "orbitProperties_pass": "DESCENDING"}

    Returns:
        list of ee.Filter: the filters to apply to the S1 collection

    Raises:
        ValueError: if a value is not a valid mode or orbit direction
    """
    filters = []

    instrument_mode = properties.get("instrumentMode")
    if instrument_mode:
        instrument_mode = str(instrument_mode).strip().upper()
        if instrument_mode not in VALID_S1_INSTRUMENT_MODES:
            raise ValueError(
                f"Invalid Sentinel-1 instrumentMode: {instrument_mode!r}. "
                f"Valid options are: {list(VALID_S1_INSTRUMENT_MODES)}"
            )
        filters.append(ee.Filter.eq("instrumentMode", instrument_mode))

    orbit_pass = properties.get("orbitProperties_pass")
    if orbit_pass:
        orbit_pass = str(orbit_pass).strip().upper()
        if orbit_pass not in VALID_S1_ORBIT_PASSES:
            raise ValueError(
                f"Invalid Sentinel-1 orbitProperties_pass: {orbit_pass!r}. "
                f"Valid options are: {list(VALID_S1_ORBIT_PASSES)}"
            )
        filters.append(ee.Filter.eq("orbitProperties_pass", orbit_pass))

    return filters


def get_tier1_images(inputs, polygon, dates, scene_cloud_cover, months_list):
    """
    Retrieves Tier 1 images from Landsat and Sentinel collections based on the given inputs.
    Calls GEE's get_image_info function to get the images.

    Args:
        inputs (dict): A dictionary containing input parameters.
            1. sat_list (list): List of satellite names to filter the images. Eg. ['L5', 'L7', 'L8', 'L9', 'S2', 'S1']
            2. S2tile (str): Sentinel-2 tile name to filter the images.
            3. sentinel_1_properties (dict): Properties for Sentinel-1 images, including polarization and instrument mode.
            4. landsat_collection (str): Landsat collection name, default is 'C02'.
            5. min_roi_coverage (float): Minimum coverage percentage for the images, default is 0.50.
        polygon (str): The polygon representing the area of interest.
        dates (list): A list of dates to filter the images.
        scene_cloud_cover (float): The maximum cloud cover percentage allowed for the images.
        months_list (list): A list of months to filter the images.

    Returns:
        dict: A dictionary containing the retrieved images, grouped by satellite name.
        Example:
            {
                'L5': [image1, image2, ...],
                'L7': [image1, image2, ...],
                'L8': [image1, image2, ...],
                'L9': [image1, image2, ...],
                'S2': [image1, image2, ...],
                'S1': [image1, image2, ...],
            }

    """
    print("- In Landsat Tier 1 & Sentinel-2 Level-1C:")
    col_names_T1 = {
        "L5": "LANDSAT/LT05/C02/T1_TOA",
        "L7": "LANDSAT/LE07/C02/T1_TOA",
        "L8": "LANDSAT/LC08/C02/T1_TOA",
        "L9": "LANDSAT/LC09/C02/T1_TOA",
        "S2": "COPERNICUS/S2_HARMONIZED",
        "S1": "COPERNICUS/S1_GRD",
    }

    im_dict_T1 = dict([])
    for satname in inputs["sat_list"]:
        im_dict_T1[satname] = []
        if satname == "S1":
            # One query for all the requested polarizations, AND-filtered so that every
            # scene returned carries all of them. Querying once per polarization would
            # return each dual-pol scene several times over.
            polarizations = get_sentinel1_polarizations(inputs)
            # user values override the defaults key by key; instrumentMode and
            # orbitProperties_pass become metadata filters on the collection
            s1_properties = {
                **DEFAULT_SENTINEL_1_PROPERTIES,
                **(inputs.get("sentinel_1_properties") or {}),
            }
            im_list = get_image_info(
                col_names_T1[satname],
                satname,
                polygon,
                dates,
                S2tile=inputs.get("S2tile", ""),
                scene_cloud_cover=scene_cloud_cover,
                months_list=months_list,
                polar=polarizations,
                sentinel_1_properties=s1_properties,
                min_roi_coverage=inputs.get("min_roi_coverage", 0.30),
            )
            im_dict_T1[satname] = im_list
            print(f"     {satname} {'+'.join(polarizations)}: {len(im_list)} images")
        else:
            # get the list of images available for the particular satellite at the location given by the polygon and the dates
            im_list = get_image_info(
                col_names_T1[satname],
                satname,
                polygon,
                dates,
                S2tile=inputs.get("S2tile", ""),
                scene_cloud_cover=scene_cloud_cover,
                months_list=months_list,
                polar=None,
                min_roi_coverage=inputs.get("min_roi_coverage", 0.30),
            )

            if satname == "S2":
                im_list = filter_S2_collection(im_list)
            print("     %s: %d images" % (satname, len(im_list)))
            im_dict_T1[satname] = im_list
    return im_dict_T1


def remove_existing_images_if_needed(inputs, im_dict_T1):
    """
    Removes existing images if needed based on the provided inputs.

    Args:
        inputs (dict): A dictionary containing the input parameters.
        im_dict_T1 (dict): A dictionary containing the imagery data.

    Returns:
        dict: The updated imagery data dictionary after removing existing images if needed.
    """
    filepath = os.path.join(inputs["filepath"], inputs["sitename"])
    if os.path.exists(filepath):
        sat_list = inputs["sat_list"]
        metadata = get_metadata(inputs)
        im_dict_T1 = remove_existing_imagery(im_dict_T1, metadata, sat_list, inputs)
    return im_dict_T1


def get_tier2_images(inputs, polygon, dates_str, scene_cloud_cover, months_list):
    """
    Retrieves Tier 2 images for Landsat satellites.

    Args:
        inputs (dict): A dictionary containing input parameters.
            1. sat_list (list): List of satellite names to filter the images. Eg. ['L5', 'L7', 'L8', 'L9', 'S2', 'S1']
            2. S2tile (str): Sentinel-2 tile name to filter the images.
            3. sentinel_1_properties (dict): Properties for Sentinel-1 images, including polarization and instrument mode.
            4. landsat_collection (str): Landsat collection name, default is 'C02'.
            5. min_roi_coverage (float): Minimum coverage percentage for the images, default is 0.50.
        polygon (str): The polygon coordinates of the area of interest.
        dates_str (str): A string representing the dates of interest.
        scene_cloud_cover (float): The maximum allowable cloud cover for the scenes.
        months_list (list): A list of months to filter the images.

    Returns:
        dict: A dictionary containing the retrieved Tier 2 images for each Landsat satellite.
        Example:
            {
                'L5': [image1, image2, ...],
                'L7': [image1, image2, ...],
                'L8': [image1, image2, ...],
                'L9': [image1, image2, ...],
                'S2': [image1, image2, ...],
                'S1': [image1, image2, ...],
            }


    """
    print("- In Landsat Tier 2 (not suitable for time-series analysis):", end="\n")
    col_names_T2 = {
        "L5": "LANDSAT/LT05/C02/T2_TOA",
        "L7": "LANDSAT/LE07/C02/T2_TOA",
        "L8": "LANDSAT/LC08/C02/T2_TOA",
    }
    im_dict_T2 = dict([])
    for satname in inputs["sat_list"]:
        if satname not in col_names_T2:
            continue  # only L5/L7/L8 have a Tier 2 collection (not L9, S2 or S1)
        im_list = get_image_info(
            col_names_T2[satname],
            satname,
            polygon,
            dates_str,
            S2tile=inputs.get("S2tile", ""),
            scene_cloud_cover=scene_cloud_cover,
            months_list=months_list,
            min_roi_coverage=inputs.get("min_roi_coverage", 0.30),
        )
        print("     %s: %d images" % (satname, len(im_list)))
        im_dict_T2[satname] = im_list
    return im_dict_T2


def check_dates_order(dates):
    """
    Check if the given list of dates is in the correct chronological order.

    Args:
        dates (list): A list of dates.

    Raises:
        Exception: If the dates are not in the correct chronological order.

    """
    if dates[1] <= dates[0]:
        raise Exception("Verify that your dates are in the correct chronological order")


def check_if_ee_initialized():
    """
    Checks if the Earth Engine API is initialized.

    Returns:
        bool: True if the Earth Engine API is initialized, False otherwise.
    """
    try:
        ee.ImageCollection("LANDSAT/LT05/C02/T1_TOA")
        return True
    except Exception as e:
        print(
            f"Earth Engine API is not initialized: {e} Run authenticate_and_initialize(project=<project_id>) to initialize access to Google Earth Engine."
        )
        return False


def validate_collection(inputs: dict):
    """
    Validates the Landsat collection specified in the inputs dictionary is C02

    Args:
        inputs (dict): A dictionary containing the input parameters.
        Should contain the key 'landsat_collection' with the Landsat collection name.

    Returns:
        dict: The updated inputs dictionary with the Landsat collection validated.

    Raises:
        ValueError: If the Landsat collection is invalid.
    """
    if inputs.get("landsat_collection") == "C01":
        print(
            f"Google has deprecated the C01 collection, switching to C02.\n Learn more: https://developers.google.com/earth-engine/landsat_c1_to_c2"
        )
        # change the inputs settings to C02
        inputs["landsat_collection"] = "C02"
    elif inputs.get("landsat_collection") != "C02":
        raise ValueError(
            f"Invalid Landsat collection: {inputs.get('landsat_collection')}. Choose 'C02'."
        )
    return inputs


def filter_images_by_month(im_list, satname, months_list, **kwargs):
    """
    Removes from the EE collection very cloudy images (>95% cloud cover)

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection
    satname:
        name of the satellite mission
    months_list: list
        list of months to keep

    Returns:
    -----------
    im_list_upt: list
        updated list of images
    """
    if not months_list:
        return im_list

    if satname in ["L5", "L7", "L8", "L9"]:
        property_name = "DATE_ACQUIRED"
        # get the properties of the images
        img_properties = [_["properties"][property_name] for _ in im_list]
        img_months = [
            datetime.strptime(img["properties"][property_name], "%Y-%m-%d").month
            for img in im_list
        ]
        if np.any([img_month not in months_list for img_month in img_months]):
            # drop all the images that are not in the months_list
            idx_delete = np.where(
                [
                    datetime.strptime(date_acquired, "%Y-%m-%d").month
                    not in months_list
                    for date_acquired in img_properties
                ]
            )[0]
            im_list_upt = [x for k, x in enumerate(im_list) if k not in idx_delete]
        else:
            im_list_upt = im_list
    elif satname in ["S1", "S2"]:
        property_name = "system:time_start"
        img_properties = [_["properties"][property_name] for _ in im_list]
        img_months = [
            datetime.fromtimestamp(img["properties"][property_name] / 1000.0).month
            for img in im_list
        ]
        if np.any([img_month not in months_list for img_month in img_months]):
            idx_delete = np.where(
                [
                    datetime.fromtimestamp(date_acquired / 1000.0).month
                    not in months_list
                    for date_acquired in img_properties
                ]
            )[0]
            im_list_upt = [x for k, x in enumerate(im_list) if k not in idx_delete]
        else:
            im_list_upt = im_list

    return im_list_upt


def filter_collection_by_coverage(
    collection: ee.collection,
    polygon: ee.Geometry,
    min_roi_coverage=0.3,
    tol: float = 1000,
    **kwargs,
) -> ee.collection:
    """
    Filters an Earth Engine image collection to retain only images that sufficiently cover a given polygon.
    This function calculates the fraction of the polygon's area covered by each image's footprint and filters out images that do not meet the specified minimum coverage threshold. Additional properties, such as the coverage fraction, image footprint, and overlap geometry, are added to each image.
    Args:
        collection (ee.collection): The Earth Engine image collection to filter.
        polygon (ee.Geometry): The polygon geometry to assess coverage against.(Note this is the ROI)
        min_roi_coverage (float, optional): Minimum required fraction (0-1) of the polygon area that must be covered by an image. Defaults to 0.3.
        tol (float, optional): The error tolerance (in meters) for area and intersection calculations. Defaults to 1000.
    Returns:
        ee.collection: The image collection filtered to only include images that covered at least min_roi_coverage of the polygon, with each image annotated with coverage information.
    """
    # Assume that filtering by date and polygon coverage has already occured

    polygon_area = polygon.area(tol)

    def add_coverage_fraction(img: ee.Image):
        footprint = img.geometry()
        overlap = footprint.intersection(polygon, tol)
        overlap_area = overlap.area(tol)
        coverage_frac = overlap_area.divide(polygon_area)
        return img.set(
            {
                "coverage_frac": coverage_frac,
                "footprint_geom": footprint,
                "overlap_geom": overlap,
            }
        )

    # Add new properties to each image in the collection
    with_coverage = collection.map(add_coverage_fraction)
    # Filter out any images in the collection that did not not cover at least min_roi_coverage of the polygon
    # Note this is to prevent us from downloading teeny tiny clips
    filtered_collection = with_coverage.filter(
        ee.Filter.gte("coverage_frac", min_roi_coverage)
    )
    return filtered_collection


@retry  # Apply the retry decorator to the function
def get_image_info(
    collection,
    satname,
    polygon,
    dates,
    scene_cloud_cover: float = 0.95,
    min_roi_coverage: float = 0.3,
    **kwargs,
):
    """
    Reads info about EE images for the specified collection, satellite, and dates
    Modified by Sharon Batiste to make this function compatible with CoastSeg

    Arguments:
    -----------
    collection: str
        name of the collection (e.g. 'LANDSAT/LC08/C02/T1_TOA')
    satname: str
        name of the satellite mission
    polygon: list
        coordinates of the polygon in lat/lon
    dates: list of str
        start and end dates (e.g. '2022-01-01')
    scene_cloud_cover: float (default: 0.95)
        maximum cloud cover percentage for the scene (not just the ROI)
    min_roi_coverage: float (default: 0.3)
        minimum required fraction of the polygon area that must be covered by an image
    kwargs: dict
        additional arguments to pass to the function (e.g. S2tile, polar)
        polar: list of polarizations to filter by (e.g. ['HH', 'VH'])
        S2tile: Sentinel-2 tile name to filter the images (e.g. '58GGP')
        sentinel_1_properties: dict of S1 metadata to filter by, see
            create_sentinel1_metadata_filters (instrumentMode, orbitProperties_pass)

    Returns:
    -----------
    im_list: list of ee.Image objects
        list with the info for the images
    """
    start_date, end_date = _normalize_dates(dates)

    # get info about images
    ee_col = ee.ImageCollection(collection)
    # Initialize the collection with filterBounds and filterDate
    ee_col = ee_col.filterBounds(ee.Geometry.Polygon(polygon)).filterDate(
        start_date, end_date
    )

    # If "polar" is included it contains a list of polarizations to filter by. Example: ['VV', 'VH']
    # Only scenes carrying ALL of them are kept, so every image can supply the full set.
    if kwargs.get("polar"):
        polar_filter = create_polarization_filter(kwargs["polar"], mode="and")
        ee_col = ee_col.filter(polar_filter)

    # Sentinel-1 scene metadata filters: instrumentMode and orbitProperties_pass
    for metadata_filter in create_sentinel1_metadata_filters(
        kwargs.get("sentinel_1_properties") or {}
    ):
        ee_col = ee_col.filter(metadata_filter)

    # If "S2tile" key is in kwargs and its associated value is truthy (not an empty string, None, etc.),
    # then apply an additional filter to the collection.
    if kwargs.get("S2tile"):
        ee_col = ee_col.filterMetadata("MGRS_TILE", "equals", kwargs["S2tile"])  # 58GGP
        print(f"Only keeping user-defined S2tile: {kwargs['S2tile']}")

    # This splits the im_list to avoid the 5000 image limit by making multiple orders within the list
    im_list = get_images_list_from_collection(
        ee_col, polygon, dates, min_roi_coverage=min_roi_coverage
    )

    # remove very cloudy images (>95% cloud cover)
    im_list = remove_cloudy_images(
        im_list, satname, cloud_threshold=scene_cloud_cover, **kwargs
    )
    im_list = filter_images_by_month(
        im_list,
        satname,
        kwargs.get(
            "months_list",
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
            ],
        ),
    )
    return im_list


@retry
def remove_dimensions_from_bands(image_ee, **kwargs):
    # first delete dimensions key from dictionary
    # otherwise the entire image is extracted (don't know why)
    im_bands = image_ee.getInfo()["bands"]

    # remove some additional masks provided with S2
    im_bands = [band for band in im_bands if "MSK_CLASSI" not in band["id"]]
    for j in range(len(im_bands)):
        if "dimensions" in im_bands[j]:
            del im_bands[j]["dimensions"]
    return im_bands


def get_image_quality(satname: str, im_meta: dict) -> str:
    """
    Get the image quality for a given satellite name and image metadata.
    If the image quality field is missing 'NA' is returned

    Args:
        satname (str): Satellite name identifier (e.g. 'L5', 'L7', 'L8', 'L9', 'S2')
        im_meta (dict): Image metadata containing the properties field

    Returns:
        im_quality (str): Image quality value from the metadata based on satellite name

    Satellite information:
        L5, L7: Landsat 5 and Landsat 7
        L8, L9: Landsat 8 and Landsat 9
        S2: Sentinel-2
    """
    default_quality = "PASSED"  # Default value if quality is not found

    # Mapping of satellite names to their respective quality keys
    quality_keys = {
        "L5": "IMAGE_QUALITY",
        "L7": "IMAGE_QUALITY",
        "L8": "IMAGE_QUALITY_OLI",
        "L9": "IMAGE_QUALITY_OLI",
    }
    if satname in quality_keys:
        return im_meta["properties"].get(quality_keys[satname], default_quality)
    elif satname == "S2":
        # List of possible quality flag names for Sentinel-2
        flag_names = ["RADIOMETRIC_QUALITY", "RADIOMETRIC_QUALITY_FLAG"]
        # Try to find the first existing quality flag in the metadata properties
        for flag in flag_names:
            if flag in im_meta["properties"]:
                return im_meta["properties"][flag]
        # If no quality flag is found, log a warning and default to "PASSED"
        print(
            "WARNING: could not find Sentinel-2 geometric quality flag,"
            + " raise an issue at https://github.com/kvos/CoastSat/issues"
            + " and add your inputs in text (not a screenshot, please)."
        )
        return "PASSED"
    else:
        return default_quality  # Return default value for unknown satellites


def get_georeference_accuracy(satname: str, im_meta: dict) -> str:
    """
    Get the accuracy of geometric reference based on the satellite name and
    the image metadata.

    Landsat default value of accuracy is RMSE = 12m

    Sentinel-2 don't provide a georeferencing accuracy (RMSE as in Landsat), instead the images include a quality control flag in the metadata that
    indicates georeferencing accuracy: a value of 1 signifies a passed geometric check, while -1 denotes failure(i.e., the georeferencing is not accurate).
    This flag is stored in the image's metadata, which is additional information about the image stored with it.
    However, the specific property or field in the metadata where this flag is stored can vary across the Sentinel-2 archive,
    meaning it's not always in the same place or under the same name.

    Parameters:
    satname (str): Satellite name, e.g., 'L5', 'L7', 'L8', 'L9', or 'S2'.
    im_meta (dict): Image metadata containing the properties for geometric
                    reference accuracy.

    Returns:
    str: The accuracy of the geometric reference.
    """
    acc_georef = "PASSED"
    if satname in ["L5", "L7", "L8", "L9"]:
        # average georefencing error across Landsat collection (RMSE = 12m)
        acc_georef = im_meta["properties"].get("GEOMETRIC_RMSE_MODEL", 12)
    elif satname == "S2":
        flag_names = [
            "GEOMETRIC_QUALITY_FLAG",
            "GEOMETRIC_QUALITY",
            "quality_check",
            "GENERAL_QUALITY_FLAG",
        ]
        for flag in flag_names:
            if flag in im_meta["properties"]:
                return im_meta["properties"][flag]
        # if none fo the flags appeared then set the acc_georef to passed
        acc_georef = "PASSED"
    return acc_georef


def handle_duplicate_image_names(
    all_names: List[str],
    bands: Dict[str, str],
    im_fn: Dict[str, str],
    im_date: str,
    satname: str,
    sitename: str,
    suffix: str,
) -> Dict[str, str]:
    """
    This function handles duplicate names for image files. If an image file name is already in use,
    it adds a '_dupX' suffix to the name (where 'X' is a counter for the number of duplicates).

    Parameters:
    all_names (list): A list containing all image file names that have been handled so far.
    bands (dict): A dictionary where the keys are the band names.
    im_fn (dict): A dictionary where the keys are the band names and the values are the current file names for each band.
    im_date (str): A string representing the date when the image was taken.
    satname (str): A string representing the name of the satellite that took the image.
    sitename: A string representing name of directory containing all downloaded images
    suffix (str): A string representing the file extension or other suffix to be added to the file name.

    Returns:
    im_fn (dict): The updated dictionary where the keys are the band names and the values are the modified file names for each band.
    """
    # if multiple images taken at the same date add 'dupX' to the name (duplicate)
    duplicate_counter = 0
    while im_fn["ms"] in all_names:
        duplicate_counter += 1
        for key in bands.keys():
            im_fn[key] = (
                f"{im_date}_{satname}_{sitename}"
                f"_{key}_dup{duplicate_counter}{suffix}"
            )
    return im_fn


def filter_bands(bands_source, band_ids):
    """Filters the bands from bands_source based on the given band_ids."""
    return [band for band in bands_source if band["id"] in band_ids]


def merge_image_tiers(inputs, im_dict_T1, im_dict_T2):
    """
    Merges im_dict_T2 into im_dict_T1 based on the keys provided in inputs["sat_list"].
    Parameters:
    - inputs: Dictionary with "sat_list" as one of its keys containing a list of keys to check.
    - im_dict_T1: First dictionary to merge into.
    - im_dict_T2: Second dictionary containing values to be added to im_dict_T1.
    Returns:
    - Updated im_dict_T1 after merging.
    """

    # Merge tier 2 imagery into dictionary if include_T2 is True
    if inputs.get("include_T2", False):
        for key in inputs["sat_list"]:
            if key in ["S2", "L9"]:
                continue
            else:
                # Check if key exists in both dictionaries
                if key in im_dict_T1 and key in im_dict_T2:
                    im_dict_T1[key] += im_dict_T2[key]
                elif key in im_dict_T2:  # If key only exists in im_dict_T2
                    im_dict_T1[key] = im_dict_T2[key]  # Add it to im_dict_T1

    return im_dict_T1


def count_total_images(image_dict, tier=1):
    """
    Counts the total number of images available in a nested dictionary.

    The function recursively traverses the dictionary and sums the lengths of all lists found.
    It can handle dictionaries where values may be:
    - lists of images,
    - nested dictionaries with further lists,
    - or a mix of both.

    Parameters:
        image_dict (dict): A dictionary where keys represent satellite names and
                           values are either lists of images or nested dictionaries
                           containing lists.

    Returns:
        int: Total number of images across all satellite sources.

    Example:
        im_dict_T1 = {
            'S1': [1, 2, 3, 4],
            'S2': [9, 10, 11],
        }

        count_total_images(im_dict_T1)
        # Output:
        #   Total images available to download from Tier 1: 11 images
    """

    def recursive_count(d):
        total = 0
        if isinstance(d, list):
            return len(d)
        elif not isinstance(d, dict):
            return 0
        for value in d.values():
            if isinstance(value, list):
                total += len(value)
            elif isinstance(value, dict):
                total += recursive_count(value)
            else:
                return 0  # In case there are non-list, non-dict values
        return total

    total_images = sum(recursive_count(subdict) for subdict in image_dict.values())
    print(
        f"  Total images available to download from Tier {tier}: %d images"
        % total_images
    )
    return total_images


def check_images_available(
    inputs: dict,
    months_list: list[int] = None,
    scene_cloud_cover: float = 0.95,
    tier1: bool = True,
    tier2: bool = False,
    min_roi_coverage: float = 0.3,
    **kwargs,
) -> tuple[list[dict], list[dict]]:
    """
    Scan the GEE collections to see how many images are available for each
    satellite mission (L5,L7,L8,L9,S2,S1), collection (C02) and tier (T1,T2).

    Note: Landsat Collection 1 (C01) is deprecated. Users should migrate to Collection 2 (C02).
    For more information, visit: https://developers.google.com/earth-engine/landsat_c1_to_c2

    Before running this function, make sure to run authenticate_and_initialize(project=<project_id>). Otherwise, the function will raise an error.

    KV WRL 2018

    Arguments:
    -----------
    inputs: dict
        inputs dictionary
    months_list: list of int
        list of months to filter the images by and only keep images in these months (default is None)
        example: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    scene_cloud_cover: int
        maximum cloud cover percentage for the scene (default is 95)
        Note: this is the entire scene not just the ROI
    tier1: bool
        whether to include Tier 1 images (default is True)
    tier2: bool
        whether to include Tier 2 images (default is False)
    min_roi_coverage: float
        minimum required fraction of the ROI area that must be covered by an image (default is 0.3)
    kwargs: dict
        additional arguments to pass to the function (e.g. S2tile, polar)
        polar: list of polarizations to filter by (e.g. ['HH', 'VH'])
        S2tile: Sentinel-2 tile name to filter the images (e.g. '58GGP')

    Returns:
    -----------
    im_dict_T1: dict
        dictionary with the info for the Tier 1 images
        example: {'L5': [image1, image2, ...], 'L7': [image1, image2, ...], ...}
    im_dict_T2: dict
        dictionary with the info for the Tier 2 images
        example: {'L5': [image1, image2, ...], 'L7': [image1, image2, ...], ...}

    Raises:
        Exception: If the Earth Engine API is not initialized.
        Exception: If the dates in inputs["dates"] are not in chronological order.
    """
    if months_list is None:
        months_list = list(range(1, 13))

    dates = [datetime.strptime(_, "%Y-%m-%d") for _ in inputs["dates"]]
    dates_str = inputs["dates"]
    polygon = inputs["polygon"]

    # add min_roi_coverage to inputs dictionary so that used in get_tier2_images and get_tier1_images
    inputs["min_roi_coverage"] = min_roi_coverage

    im_dict_T2 = {}
    im_dict_T1 = {}

    check_dates_order(dates)

    if check_if_ee_initialized() == False:
        raise Exception(
            "Earth Engine API is not initialized. Please initialize it first with ee.Initialize(project=<project_id>)"
        )

    print(
        "Number of images available between %s and %s:" % (dates_str[0], dates_str[1]),
        end="\n",
    )

    if tier1:
        im_dict_T1 = get_tier1_images(
            inputs, polygon, dates, scene_cloud_cover, months_list
        )
        im_dict_T1 = remove_existing_images_if_needed(inputs, im_dict_T1)

    count_total_images(im_dict_T1, tier=1)

    # S2 does not have a tier 2 collection so if it was the only satellite requested, return
    if len(inputs["sat_list"]) == 1 and inputs["sat_list"][0] == "S2":
        return im_dict_T1, []

    if tier2:
        im_dict_T2 = get_tier2_images(
            inputs, polygon, dates_str, scene_cloud_cover, months_list
        )
        im_dict_T2 = remove_existing_images_if_needed(inputs, im_dict_T2)

    count_total_images(im_dict_T2, tier=2)

    return im_dict_T1, im_dict_T2


def download_S1_image(image_ee, polygon, polarization, save_path, **kwargs):
    """
    Downloads a single polarization of a Sentinel-1 scene as a single-band .tif.

    Args:
        image_ee (ee.Image): the Sentinel-1 image.
        polygon (list): the ROI to crop to.
        polarization (str): the polarization band to download, e.g. 'VH'.
        save_path (str): folder to download into.
        **kwargs: forwarded to download_tif (image_id and logger, used by @retry).

    Returns:
        str: path to the downloaded .tif.
    """
    # Get region and download. The ROI is snapped to this band's native pixel grid so
    # no resampling happens; all polarizations of a GRD scene share that grid.
    proj = image_ee.select(polarization).projection()
    region = adjust_polygon(polygon, proj, **kwargs)
    return download_tif(image_ee, region, polarization, save_path, "S1", **kwargs)


def process_sentinel1_image(
    image_ee,
    im_meta,
    inputs,
    filepaths,
    im_date,
    suffix,
    all_names,
    logger,
    default_sentinel_1_properties=None,
):
    """
    Processes a single Sentinel-1 image: downloads every requested polarization,
    renames them, and saves one metadata file for the scene.

    Each polarization is saved as its own single-band .tif in its own folder
    (S1/VV/, S1/VH/, ...). The files of a scene share a name apart from the
    polarization token, which is how the read path pairs them back up.

    Returns:
        dict: Contains 'skip_image', and on success 'filename_ms', 'im_fn' and
        'band_order' (the polarizations downloaded, in canonical order).
    """
    try:
        polarizations = get_sentinel1_polarizations(
            inputs, default_sentinel_1_properties, logger=logger
        )

        # Require every requested polarization: a scene missing one is skipped entirely
        # rather than downloaded as a partial set.
        bands_by_id = {band.get("id"): band for band in im_meta.get("bands", [])}
        missing = [polar for polar in polarizations if polar not in bands_by_id]
        if missing:
            logger.warning(
                f"Skipping Sentinel-1 image {im_meta.get('id', 'unknown')}: missing "
                f"polarization(s) {missing}; scene carries {sorted(bands_by_id)}"
            )
            return {"skip_image": True}

        # the polarizations of a scene must share a pixel grid or the stack would be
        # misaligned when they are read back together
        grids = {
            (
                bands_by_id[polar].get("crs"),
                tuple(bands_by_id[polar].get("crs_transform", ())),
            )
            for polar in polarizations
        }
        if len(grids) > 1:
            logger.warning(
                f"Sentinel-1 image {im_meta.get('id', 'unknown')}: polarizations "
                f"{polarizations} are not on the same pixel grid, the stacked bands "
                "may be misaligned."
            )

        im_epsg = int(bands_by_id[polarizations[0]]["crs"][5:])
        # map polarization -> its folder; filepaths is [meta, <polar1>, <polar2>, ...]
        fp_by_polar = {os.path.basename(fp): fp for fp in filepaths[1:]}

        downloaded = []  # (polar, filename, path) for each polarization written
        try:
            for polar in polarizations:
                fp = fp_by_polar[polar]
                default_tif_path = download_S1_image(
                    image_ee,
                    inputs["polygon"],
                    polar,
                    fp,
                    image_id=im_meta.get("id"),
                    logger=logger,
                )

                # Create safe filename (get_file_name adds it to all_names itself)
                filename = get_file_name(
                    im_date, "S1", inputs["sitename"], polar, suffix, all_names
                )
                new_filepath = os.path.join(fp, filename)
                if os.path.exists(new_filepath):
                    os.remove(new_filepath)  # Remove the existing file before renaming
                os.rename(default_tif_path, new_filepath)
                downloaded.append((polar, filename, new_filepath))
        except Exception:
            # never leave a scene with only some of its polarizations on disk
            for _, _, path in downloaded:
                for stale in (path, path + ".aux.xml"):
                    if os.path.exists(stale):
                        os.remove(stale)
            raise

        # One metadata file per scene, named like the optical ones. Writing one per
        # polarization would make get_metadata() list every scene more than once.
        primary_polar, primary_filename, primary_path = downloaded[0]
        width, height = SDS_tools.get_image_dimensions(primary_path)

        filename_txt = primary_filename.replace(f"_{primary_polar}", "").replace(
            ".tif", ""
        )
        metadict = {
            "filename": primary_filename,
            "epsg": im_epsg,
            "im_width": width,
            "im_height": height,
            # polarizations downloaded for this scene, in the order they are stacked
            "band_order": polarizations,
            "orbitProperties_pass": im_meta["properties"]["orbitProperties_pass"],
            "transmitterReceiverPolarisation": im_meta["properties"][
                "transmitterReceiverPolarisation"
            ],
            "resolution": im_meta["properties"]["resolution"],
            "resolution_meters": im_meta["properties"]["resolution_meters"],
            "instrumentMode": im_meta["properties"]["instrumentMode"],
        }
        write_metadata_file(filepaths[0], filename_txt, metadict)

        logger.info(
            f"Downloaded Sentinel-1 image {im_meta.get('id', 'unknown')} as "
            f"{filename_txt} with polarizations {polarizations}"
        )

        return {
            "skip_image": False,
            "filename_ms": primary_filename,
            "im_fn": {"ms": primary_filename},  # Maintain consistency
            "band_order": polarizations,
        }
    except Exception as e:
        logger.error(
            f"Failed to process Sentinel-1 image {im_meta.get('id', 'unknown')}.\n{e}"
        )
        return {"skip_image": True}


def retrieve_images(
    inputs,
    cloud_threshold: float = 0.95,
    cloud_mask_issue: bool = False,
    save_jpg: bool = True,
    apply_cloud_mask: bool = True,
    months_list: Optional[List[int]] = None,
    max_cloud_no_data_cover: float = 0.95,
    scene_cloud_cover: float = 0.95,
    min_roi_coverage: float = 0.3,
    s2cloudless_prob: int = 60,
    skip_cache_flush_interval: int = 5,
):
    """
    Downloads all images from Landsat 5, Landsat 7, Landsat 8, Landsat 9 and Sentinel-2
    covering the area of interest and acquired between the specified dates.
    The downloaded images are in .TIF format and organized in subfolders, divided
    by satellite mission. The bands are also subdivided by pixel resolution.

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
            sat_list = ['L5', 'L7', 'L8', 'S2']
            ```
        'filepath_data': str
            filepath to the directory where the images are downloaded

    cloud_threshold: float (default: 0.95)
        maximum cloud cover percentage within the ROI for images to be kept, otherwise removed.
    cloud_mask_issue: bool (default: False)
        Make True is one of the satellites is mis-identifying sand as clouds
    save_jpg: bool:
        save jpgs for each image downloaded
    apply_cloud_mask: bool (default: True)
        apply cloud mask to the images. If False, the images are not cloud masked, but the mask is still generated.
    months_list: list of int (default: None)
        Any images within these months are kept, all others are removed
        eg. [1,2,3,4,5,6,7,8,9,10,11,12] for all months
    max_cloud_no_data_cover: float (default: 0.95)
        maximum cloud cover & no data combined percentage within the ROI for images to be kept, otherwise removed
    scene_cloud_cover: float (default: 0.95)
        maximum cloud cover percentage for the scene (not just the ROI)
    min_roi_coverage: float (default: 0.3)
        The minimum percentage of the ROI that must be covered by the image.
    s2cloudless_prob: int (default: 60)
        Threshold used for Sentinel-2 s2cloudless cloud masking when filtering cloud/no-data.
    skip_cache_flush_interval: int (default: 10)
        Number of processed images between skip-cache disk flushes when cache changes are pending.

    Notes:
    -----------
    Cloud/no-data skips are cached in a site-level JSON file so resumed runs can avoid
    re-downloading known rejected images when they still exceed the current thresholds.
    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # create a new directory for this site with the name of the site
    im_folder = os.path.join(inputs["filepath"], inputs["sitename"])
    if not os.path.exists(im_folder):
        os.makedirs(im_folder)
    # Initialize the logger
    logger = setup_logger(im_folder)

    if check_if_ee_initialized() == False:
        raise Exception(
            "Earth Engine API is not initialized. Please initialize it first with ee.Initialize(project=<project_id>)"
        )

    # add the min_roi_coverage to the inputs dictionary
    inputs["min_roi_coverage"] = min_roi_coverage

    # validates the inputs have references the correct collection (C02)
    inputs = validate_collection(inputs)

    # check image availabiliy and retrieve list of images
    im_dict_T1, im_dict_T2 = check_images_available(
        inputs, months_list, scene_cloud_cover, min_roi_coverage=min_roi_coverage
    )

    # merge the two image collections tiers into a single dictionary
    im_dict_T1 = merge_image_tiers(inputs, im_dict_T1, im_dict_T2)

    # remove UTM duplicates in S2 collections (they provide several projections for same images)
    if "S2" in inputs["sat_list"] and len(im_dict_T1["S2"]) > 0:
        im_dict_T1["S2"] = filter_S2_collection(im_dict_T1["S2"])
        # get s2cloudless collection
        im_dict_s2cloudless = get_s2cloudless(im_dict_T1["S2"], inputs)

    # QA band for each satellite mission
    qa_band_Landsat = "QA_PIXEL"
    qa_band_S2 = "QA60"

    # the cloud mask band for Sentinel-2 images is the s2cloudless probability
    bands_dict = {
        "L5": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
        "L7": ["B1", "B2", "B3", "B4", "B5", qa_band_Landsat],
        "L8": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
        "L9": ["B2", "B3", "B4", "B5", "B6", qa_band_Landsat],
        "S2": ["B2", "B3", "B4", "B8", "s2cloudless", "B11", qa_band_S2],
        # S1 bands come from inputs['sentinel_1_properties'], not from this table
        "S1": [],
    }

    sat_list = inputs["sat_list"]
    dates = inputs["dates"]
    skip_cache = load_skip_cache(inputs)
    update_skip_cache_metadata(
        skip_cache,
        inputs,
        max_cloud_no_data_cover=max_cloud_no_data_cover,
        max_cloud_cover=cloud_threshold,
        s2cloudless_prob=s2cloudless_prob,
        cloud_mask_issue=cloud_mask_issue,
    )
    query_signature = get_skip_query_signature(inputs)
    removed_cache_entries = prune_skip_cache_for_current_thresholds(
        skip_cache,
        query_signature,
        max_cloud_no_data_cover=max_cloud_no_data_cover,
        max_cloud_cover=cloud_threshold,
    )
    # create/update cache file before downloads start so progress survives interruptions
    save_skip_cache(inputs, skip_cache)
    skip_cache_dirty = False  # flag to track if cache has pending changes that need to be flushed to disk
    processed_images_since_flush = 0  # counter to track how many images have been processed since the last cache flush

    def flush_skip_cache_if_needed(force: bool = False):
        nonlocal skip_cache_dirty, processed_images_since_flush
        if skip_cache_flush_interval <= 0:
            interval_reached = True
        else:
            interval_reached = processed_images_since_flush >= skip_cache_flush_interval

        if skip_cache_dirty and (force or interval_reached):
            save_skip_cache(inputs, skip_cache)
            skip_cache_dirty = False  # reset dirty flag after flush
            processed_images_since_flush = 0  # reset counter after flush

    if removed_cache_entries > 0:
        print(
            f"Removed {removed_cache_entries} stale skip-cache entries for current thresholds."
        )

    if np.all([len(im_dict_T1[satname]) == 0 for satname in im_dict_T1.keys()]):
        print(
            f"{inputs['sitename']}: No images to download for {sat_list} during {dates} for {cloud_threshold}% cloud cover"
        )
    else:
        # main loop to download the images for each satellite mission
        # print('\nDownloading images:')
        suffix = ".tif"
        count = 1
        num_satellites = len(im_dict_T1.keys())
        for satname in tqdm(
            im_dict_T1.keys(),
            desc=f"{inputs['sitename']}: Downloading Imagery for {num_satellites} satellites",
        ):
            count += 1
            # create subfolder structure to store the different bands
            # (for S1 that is one folder per requested polarization)
            polars = get_sentinel1_polarizations(inputs) if satname == "S1" else None
            filepaths = SDS_tools.create_folder_structure(im_folder, satname, polars)
            # initialise variables and loop through images

            bands_id = bands_dict[satname]
            all_names = []  # list for detecting duplicates
            if len(im_dict_T1[satname]) == 0:
                print(f"{inputs['sitename']}: No images to download for {satname}")
                continue
            # loop through each image
            pbar = tqdm(
                range(len(im_dict_T1[satname])),
                desc=f"{inputs['sitename']}: Downloading Imagery for {satname}",
                leave=True,
            )
            for i in pbar:
                try:
                    skip_image = False
                    # initalize the variables
                    # filepath (fp) for the multispectural file
                    fp_ms = ""
                    # store the bands availble
                    bands = dict([])
                    # dictionary containing the filepaths for each type of file downloaded
                    im_fn = dict([])

                    # get image metadata
                    im_meta = im_dict_T1[satname][i]
                    cache_key = build_skip_cache_key(im_meta, satname)
                    cache_entry = skip_cache.get("entries", {}).get(cache_key)

                    if should_skip_from_cache(
                        cache_entry,
                        query_signature,
                        max_cloud_no_data_cover=max_cloud_no_data_cover,
                        max_cloud_cover=cloud_threshold,
                    ):
                        skip_image = True
                        print(
                            f"Skipping cached image '{im_meta.get('id', 'unknown')}' due to cloud/no-data cache"
                        )
                        continue
                    elif (
                        cache_entry
                        and cache_entry.get("query_signature") == query_signature
                    ):
                        # stale entry under current thresholds; remove and allow download
                        skip_cache.get("entries", {}).pop(cache_key, None)
                        skip_cache_dirty = True

                    # get time of acquisition (UNIX time) and convert to datetime
                    acquisition_time = im_meta["properties"]["system:time_start"]
                    im_timestamp = datetime.fromtimestamp(
                        acquisition_time / 1000, tz=pytz.utc
                    )
                    im_date = im_timestamp.strftime("%Y-%m-%d-%H-%M-%S")

                    # get epsg code
                    im_epsg = int(im_meta["bands"][0]["crs"][5:])

                    # select image by id
                    image_ee = ee.Image(im_meta["id"])

                    # for S2 add s2cloudless probability band
                    if satname == "S2":
                        if len(im_dict_s2cloudless[i]) == 0:
                            skip_image = True
                            print(
                                "Warning: S2cloudless mask for image %s is not available."
                                % im_date
                            )
                            continue  # skip this image
                        im_cloud = ee.Image(im_dict_s2cloudless[i]["id"])
                        cloud_prob = im_cloud.select("probability").rename(
                            "s2cloudless"
                        )
                        image_ee = image_ee.addBands(cloud_prob)

                    if satname != "S1":
                        # get quality flags (geometric and radiometric quality)
                        accuracy_georef = get_georeference_accuracy(satname, im_meta)
                        image_quality = get_image_quality(satname, im_meta)
                        # update the loading bar with the status
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Loading bands for {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        # first delete dimensions key from dictionary
                        # otherwise the entire image is extracted (don't know why)
                        im_bands = remove_dimensions_from_bands(
                            image_ee, image_id=im_meta["id"], logger=logger
                        )
                    # =============================================================================================#
                    # Sentinel-1 download
                    # =============================================================================================#
                    if satname == "S1":

                        result = process_sentinel1_image(
                            image_ee,
                            im_meta,
                            inputs,
                            filepaths,
                            im_date,
                            suffix,
                            all_names,
                            logger,
                        )
                        if result.get("skip_image", False):
                            continue  # skip the rest of the loop if the image is not usable

                        im_fn["ms"] = result["im_fn"]["ms"]
                    # =============================================================================================#
                    # Landsat 5 download
                    # =============================================================================================#
                    if satname == "L5":
                        fp_ms = filepaths[1]
                        fp_mask = filepaths[2]
                        # select multispectral bands
                        bands["ms"] = [
                            im_bands[_]
                            for _ in range(len(im_bands))
                            if im_bands[_]["id"] in bands_id
                        ]
                        # adjust polygon to match image coordinates so that there is no resampling
                        proj = image_ee.select("B1").projection()
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: adjusting polygon {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        ee_region = adjust_polygon(
                            inputs["polygon"],
                            proj,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        # download .tif from EE (one file with ms bands and one file with QA band)
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Downloading tif for {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        fn_ms, fn_QA = download_tif(
                            image_ee,
                            ee_region,
                            bands["ms"],
                            fp_ms,
                            satname,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        # create filename for image
                        for key in bands.keys():
                            im_fn[key] = (
                                im_date
                                + "_"
                                + satname
                                + "_"
                                + inputs["sitename"]
                                + "_"
                                + key
                                + suffix
                            )
                        # if multiple images taken at the same date add 'dupX' to the name (duplicate number X)
                        im_fn = handle_duplicate_image_names(
                            all_names,
                            bands,
                            im_fn,
                            im_date,
                            satname,
                            inputs["sitename"],
                            suffix,
                        )
                        im_fn["mask"] = im_fn["ms"].replace("_ms", "_mask")
                        filename_ms = im_fn["ms"]
                        all_names.append(im_fn["ms"])

                        # resample ms bands to 15m with bilinear interpolation
                        fn_in = fn_ms
                        fn_target = fn_ms
                        fn_out = os.path.join(fp_ms, im_fn["ms"])
                        filepath_ms = os.path.join(fp_ms, im_fn["ms"])

                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Transforming {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        warp_image_to_target(
                            fn_in,
                            fn_out,
                            fn_target,
                            double_res=True,
                            resampling_method="bilinear",
                        )

                        # resample QA band to 15m with nearest-neighbour interpolation
                        fn_in = fn_QA
                        fn_target = fn_QA
                        fn_out = os.path.join(fp_mask, im_fn["mask"])
                        filepath_QA = os.path.join(fp_mask, im_fn["mask"])

                        warp_image_to_target(
                            fn_in,
                            fn_out,
                            fn_target,
                            double_res=True,
                            resampling_method="near",
                        )
                        # delete original downloads
                        for original_file in [fn_ms, fn_QA]:
                            os.remove(original_file)

                        fn = [filepath_ms, filepath_QA]
                        filter_metrics = (
                            SDS_preprocess.filter_images_by_cloud_cover_nodata(
                                fn,
                                satname,
                                cloud_mask_issue,
                                max_cloud_no_data_cover,
                                cloud_threshold,
                                do_cloud_mask=True,
                                s2cloudless_prob=s2cloudless_prob,
                                return_metrics=True,
                            )
                        )
                        skip_image = filter_metrics["filtered"]
                        # if the images was filtered out, skip the image being saved as a jpg
                        if skip_image:
                            record_skip_cache_entry(
                                skip_cache,
                                cache_key,
                                im_meta,
                                satname,
                                query_signature,
                                filter_metrics,
                            )
                            skip_cache_dirty = True
                            continue

                    # =============================================================================================#
                    # Landsat 7, 8 and 9 download
                    # =============================================================================================#
                    elif satname in ["L7", "L8", "L9"]:
                        fp_ms = filepaths[1]
                        fp_pan = filepaths[2]
                        fp_mask = filepaths[3]
                        # select bands (multispectral and panchromatic)
                        bands["ms"] = [
                            im_bands[_]
                            for _ in range(len(im_bands))
                            if im_bands[_]["id"] in bands_id
                        ]
                        bands["pan"] = [
                            im_bands[_]
                            for _ in range(len(im_bands))
                            if im_bands[_]["id"] in ["B8"]
                        ]
                        # adjust polygon for both ms and pan bands
                        proj_ms = image_ee.select("B1").projection()
                        proj_pan = image_ee.select("B8").projection()
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: adjusting polygon {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        ee_region_ms = adjust_polygon(
                            inputs["polygon"],
                            proj_ms,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        ee_region_pan = adjust_polygon(
                            inputs["polygon"],
                            proj_pan,
                            image_id=im_meta["id"],
                            logger=logger,
                        )

                        # download both ms and pan bands from EE
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Downloading tif for {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        fn_ms, fn_QA = download_tif(
                            image_ee,
                            ee_region_ms,
                            bands["ms"],
                            fp_ms,
                            satname,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        fn_pan = download_tif(
                            image_ee,
                            ee_region_pan,
                            bands["pan"],
                            fp_pan,
                            satname,
                            image_id=im_meta["id"],
                        )
                        # create filename for both images (ms and pan)
                        for key in bands.keys():
                            im_fn[key] = (
                                im_date
                                + "_"
                                + satname
                                + "_"
                                + inputs["sitename"]
                                + "_"
                                + key
                                + suffix
                            )
                        # if multiple images taken at the same date add 'dupX' to the name (duplicate number X)
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: remove duplicates for {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        im_fn = handle_duplicate_image_names(
                            all_names,
                            bands,
                            im_fn,
                            im_date,
                            satname,
                            inputs["sitename"],
                            suffix,
                        )
                        im_fn["mask"] = im_fn["ms"].replace("_ms", "_mask")
                        filename_ms = im_fn["ms"]
                        all_names.append(im_fn["ms"])

                        # resample the ms bands to the pan band with bilinear interpolation (for pan-sharpening later)
                        fn_in = fn_ms
                        fn_target = fn_pan
                        fn_out = os.path.join(fp_ms, im_fn["ms"])
                        filepath_ms = os.path.join(fp_ms, im_fn["ms"])
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Transforming {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        warp_image_to_target(
                            fn_in,
                            fn_out,
                            fn_target,
                            double_res=False,
                            resampling_method="bilinear",
                        )

                        # resample QA band to the pan band with nearest-neighbour interpolation
                        fn_in = fn_QA
                        fn_target = fn_pan
                        fn_out = os.path.join(fp_mask, im_fn["mask"])
                        filepath_QA = os.path.join(fp_mask, im_fn["mask"])
                        warp_image_to_target(
                            fn_in,
                            fn_out,
                            fn_target,
                            double_res=False,
                            resampling_method="near",
                        )

                        # rename pan band
                        try:
                            os.rename(fn_pan, os.path.join(fp_pan, im_fn["pan"]))
                        except:
                            os.remove(os.path.join(fp_pan, im_fn["pan"]))
                            os.rename(fn_pan, os.path.join(fp_pan, im_fn["pan"]))
                        # delete original downloads
                        for _ in [fn_ms, fn_QA]:
                            os.remove(_)

                        filepath_pan = os.path.join(fp_pan, im_fn["pan"])
                        fn = [filepath_ms, filepath_pan, filepath_QA]
                        filter_metrics = (
                            SDS_preprocess.filter_images_by_cloud_cover_nodata(
                                fn,
                                satname,
                                cloud_mask_issue,
                                max_cloud_no_data_cover,
                                cloud_threshold,
                                do_cloud_mask=True,
                                s2cloudless_prob=s2cloudless_prob,
                                return_metrics=True,
                            )
                        )
                        skip_image = filter_metrics["filtered"]

                        # if the images was filtered out, skip the image being saved as a jpg
                        if skip_image:
                            record_skip_cache_entry(
                                skip_cache,
                                cache_key,
                                im_meta,
                                satname,
                                query_signature,
                                filter_metrics,
                            )
                            skip_cache_dirty = True
                            continue

                        if save_jpg:
                            tif_paths = SDS_tools.get_filepath(inputs, satname)
                            SDS_preprocess.save_single_jpg(
                                filename=im_fn["ms"],
                                tif_paths=tif_paths,
                                satname=satname,
                                sitename=inputs["sitename"],
                                cloud_thresh=cloud_threshold,
                                cloud_mask_issue=cloud_mask_issue,
                                filepath_data=inputs["filepath"],
                                collection=inputs["landsat_collection"],
                                apply_cloud_mask=apply_cloud_mask,
                                s2cloudless_prob=s2cloudless_prob,
                            )

                    # =============================================================================================#
                    # Sentinel-2 download
                    # =============================================================================================#
                    elif satname in ["S2"]:
                        fp_ms = filepaths[1]
                        fp_swir = filepaths[2]
                        fp_mask = filepaths[3]
                        # select bands (10m ms RGB+NIR+s2cloudless, 20m SWIR1, 60m QA band)
                        # Assuming bands_id is a predefined list, and im_bands is a predefined source list
                        bands = {
                            "ms": filter_bands(im_bands, bands_id[:5]),
                            "swir": filter_bands(im_bands, bands_id[5:6]),
                            "mask": filter_bands(im_bands, bands_id[-1:]),
                        }
                        # adjust polygon for both ms and pan bands
                        # RGB and NIR bands 10m resolution and same footprint
                        proj_ms = image_ee.select("B1").projection()
                        # SWIR band 20m resolution and different footprint
                        proj_swir = image_ee.select("B11").projection()
                        proj_mask = image_ee.select("QA60").projection()
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: adjusting polygon {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        ee_region_ms = adjust_polygon(
                            inputs["polygon"],
                            proj_ms,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        ee_region_swir = adjust_polygon(
                            inputs["polygon"],
                            proj_swir,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        ee_region_mask = adjust_polygon(
                            inputs["polygon"],
                            proj_mask,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        # download the ms, swir and QA bands from EE
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Downloading tif for {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        fn_ms = download_tif(
                            image_ee,
                            ee_region_ms,
                            bands["ms"],
                            fp_ms,
                            satname,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        fn_swir = download_tif(
                            image_ee,
                            ee_region_swir,
                            bands["swir"],
                            fp_swir,
                            satname,
                            image_id=im_meta["id"],
                            logger=logger,
                        )
                        fn_QA = download_tif(
                            image_ee,
                            ee_region_mask,
                            bands["mask"],
                            fp_mask,
                            satname,
                            image_id=im_meta["id"],
                            logger=logger,
                        )

                        # create filename for the three images (ms, swir and mask)
                        for key in bands.keys():
                            im_fn[key] = (
                                im_date
                                + "_"
                                + satname
                                + "_"
                                + inputs["sitename"]
                                + "_"
                                + key
                                + suffix
                            )
                        # if multiple images taken at the same date add 'dupX' to the name (duplicate)
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: remove duplicates for {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        im_fn = handle_duplicate_image_names(
                            all_names,
                            bands,
                            im_fn,
                            im_date,
                            satname,
                            inputs["sitename"],
                            suffix,
                        )
                        filename_ms = im_fn["ms"]
                        all_names.append(im_fn["ms"])

                        # resample the 20m swir band to the 10m ms band with bilinear interpolation
                        fn_in = fn_swir
                        fn_target = fn_ms
                        fn_out = os.path.join(fp_swir, im_fn["swir"])
                        filepath_swir = os.path.join(fp_swir, im_fn["swir"])
                        pbar.set_description_str(
                            desc=f"{inputs['sitename']}, {satname}: Transforming {SDS_tools.ordinal(i)} image ",
                            refresh=True,
                        )
                        warp_image_to_target(
                            fn_in,
                            fn_out,
                            fn_target,
                            double_res=False,
                            resampling_method="bilinear",
                        )

                        # resample 60m QA band to the 10m ms band with nearest-neighbour interpolation
                        fn_in = fn_QA
                        fn_target = fn_ms
                        fn_out = os.path.join(fp_mask, im_fn["mask"])
                        filepath_QA = os.path.join(fp_mask, im_fn["mask"])
                        warp_image_to_target(
                            fn_in,
                            fn_out,
                            fn_target,
                            double_res=False,
                            resampling_method="near",
                        )

                        # delete original downloads
                        for _ in [fn_swir, fn_QA]:
                            os.remove(_)
                        # rename the multispectral band file
                        dst = os.path.join(fp_ms, im_fn["ms"])
                        filepath_ms = os.path.join(fp_ms, im_fn["ms"])
                        if not os.path.exists(dst):
                            os.rename(fn_ms, dst)

                        fn = [filepath_ms, filepath_swir, filepath_QA]

                        # Removes images whose cloud cover and no data coverage exceeds the threshold
                        filter_metrics = (
                            SDS_preprocess.filter_images_by_cloud_cover_nodata(
                                fn,
                                satname,
                                cloud_mask_issue,
                                max_cloud_no_data_cover,
                                cloud_threshold,
                                do_cloud_mask=True,
                                s2cloudless_prob=s2cloudless_prob,
                                return_metrics=True,
                            )
                        )
                        skip_image = filter_metrics["filtered"]

                        # if the images was filtered out, skip the image being saved as a jpg
                        if skip_image:
                            record_skip_cache_entry(
                                skip_cache,
                                cache_key,
                                im_meta,
                                satname,
                                query_signature,
                                filter_metrics,
                            )
                            skip_cache_dirty = True
                            continue

                    if save_jpg:
                        tif_paths = SDS_tools.get_filepath(inputs, satname)
                        SDS_preprocess.save_single_jpg(
                            filename=im_fn["ms"],
                            tif_paths=tif_paths,
                            satname=satname,
                            sitename=inputs["sitename"],
                            cloud_thresh=cloud_threshold,
                            cloud_mask_issue=cloud_mask_issue,
                            filepath_data=inputs["filepath"],
                            collection=inputs["landsat_collection"],
                            apply_cloud_mask=apply_cloud_mask,
                            s2cloudless_prob=s2cloudless_prob,
                        )
                except Exception as error:
                    print(
                        f"\nThe download for satellite {satname} image '{im_meta.get('id','unknown')}' failed due to {type(error).__name__ }"
                    )
                    print(error)
                    logger.error(
                        f"The download for satellite {satname} {im_meta.get('id','unknown')} failed due to \n {error} \n Traceback {traceback.format_exc()}"
                    )
                    continue
                finally:
                    try:
                        # Don't try to save the metadata if the image was skipped.
                        # attempt to save the metadata for the image even if something go wrong during the download (S1 has a different metadata format so it does not apply)
                        if satname != "S1" and not skip_image:
                            # write the metadata to a txt file if possible
                            write_image_metadata(
                                fp_ms,
                                im_fn,
                                im_meta,
                                filename_ms,
                                im_epsg,
                                accuracy_georef,
                                image_quality,
                                filepaths[0],
                                logger,
                            )
                    except Exception as e:
                        logger.error(
                            f"Could not save metadata for {im_meta.get('id','unknown')} that failed.\n{e}"
                        )
                    finally:
                        processed_images_since_flush += (
                            1  # increment the counter for processed images
                        )
                        flush_skip_cache_if_needed()

    flush_skip_cache_if_needed(force=True)
    save_skip_cache(inputs, skip_cache)
    # combines all the metadata into a single dictionary and saves it to json
    metadata = get_metadata(inputs)
    print("Satellite images downloaded from GEE and save in %s" % im_folder)
    return metadata


def parse_date_from_filename(filename: str) -> datetime:
    date_str = filename[0:19]
    return pytz.utc.localize(
        datetime(
            int(date_str[:4]),
            int(date_str[5:7]),
            int(date_str[8:10]),
            int(date_str[11:13]),
            int(date_str[14:16]),
            int(date_str[17:19]),
        )
    )


def read_metadata_file(filepath: str) -> Dict[str, Union[str, int, float]]:
    # Define keys that should always exist, with default values
    default_keys = {
        "filename": "",
        "epsg": "",
        "acc_georef": -1,  # assuming default accuracy is 'PASSED'
        "im_quality": "PASSED",  # assuming default quality is 'PASSED'
        "im_width": -1,
        "im_height": -1,
    }

    # Mapping of file key to standard field names (if necessary)
    key_mapping = {"image_quality": "im_quality"}

    metadata = default_keys.copy()

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "\t" not in line:
                continue  # Skip empty or malformed lines

            key, value = map(str.strip, line.split("\t", 1))
            key = key_mapping.get(key, key)

            # Attempt to parse value safely
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed_value = ast.literal_eval(value)
                except Exception:
                    parsed_value = value
            else:
                parsed_value = value
                # Type casting for common numeric types
                if key in ["epsg", "im_width", "im_height"]:
                    try:
                        parsed_value = int(parsed_value)
                    except ValueError:
                        try:
                            parsed_value = float(parsed_value)
                        except ValueError:
                            pass
                elif key in ["acc_georef", "im_quality"]:
                    try:
                        parsed_value = float(parsed_value)
                    except ValueError:
                        pass

            metadata[key] = parsed_value  # Add or override

    return metadata


def get_band_order_from_meta(meta_info: dict) -> list:
    """
    Returns the polarization order recorded in a Sentinel-1 metadata file.

    New metadata files carry 'band_order'. Files written before multi-polarization
    support only carry 'saved_polarization'; older ones carry neither.

    Args:
        meta_info (dict): parsed contents of a metadata .txt file.

    Returns:
        list of str: the polarizations of the scene, or [] if it recorded none.
    """
    band_order = meta_info.get("band_order")
    if isinstance(band_order, str):
        band_order = [polar.strip() for polar in band_order.split(",") if polar.strip()]
    if band_order:
        return list(band_order)
    saved_polarization = meta_info.get("saved_polarization")
    if saved_polarization:
        return [str(saved_polarization)]
    return []


def format_date(date_str: str) -> datetime:
    """
    Converts a date string to a datetime object in UTC timezone.

    Args:
        date_str (str): The date string to be converted.

    Returns:
        datetime: The converted datetime object.

    Raises:
        ValueError: If the date string is in an invalid format.
    """

    date_formats = ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]
    # convert datetime object to string
    if isinstance(date_str, datetime) == True:
        # converts the datetime object to a string
        date_str = date_str.strftime("%Y-%m-%dT%H:%M:%S")

    # format the string to a datetime object
    for date_format in date_formats:
        try:
            # creates a datetime object from a string with the date in UTC timezone
            start_date = datetime.strptime(date_str, date_format).replace(
                tzinfo=timezone.utc
            )
            return start_date
        except ValueError:
            pass
    else:
        raise ValueError(f"Invalid date format: {date_str}")


def get_metadata(inputs: dict) -> dict:
    r"""
    Gets the metadata from the downloaded images by parsing .txt files located
    in the \meta subfolder.

    KV WRL 2018
    modified by Sharon Fitzpatrick 2023

    Arguments:
    -----------
    inputs: dict with the following fields
        'sitename': str
            name of the site
        'filepath_data': str
            filepath to the directory where the images are downloaded

    Returns:
    -----------
    metadata: dict
        contains the information about the satellite images that were downloaded:
        date, filename, georeferencing accuracy and image coordinate reference system

    """
    # Construct the directory path containing the images
    filepath = os.path.join(inputs["filepath"], inputs["sitename"])
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The directory {filepath} does not exist.")
    # initialize metadata dict
    metadata = dict([])
    # loop through the satellite missions that were specified in the inputs
    satellite_list = inputs.get("sat_list", ["L5", "L7", "L8", "L9", "S2", "S1"])

    for satname in satellite_list:
        sat_path = os.path.join(filepath, satname)
        # if a folder has been created for the given satellite mission
        if satname in os.listdir(filepath):
            # update the metadata dict
            metadata[satname] = {
                "filenames": [],
                "dates": [],
                "epsg": [],
                "acc_georef": [],
                "im_quality": [],
                "im_dimensions": [],
            }
            if satname == "S1":
                # polarizations held by each scene, index-aligned with 'filenames'
                metadata[satname]["band_order"] = []
            # directory where the metadata .txt files are stored
            filepath_meta = os.path.join(sat_path, "meta")
            # get the list of filenames and sort it chronologically
            if not os.path.exists(filepath_meta):
                continue
            # Get the list of filenames and sort it chronologically
            filenames_meta = sorted(os.listdir(filepath_meta))
            # loop through the .txt files
            for im_meta in filenames_meta:
                if inputs.get("dates", None) is None:
                    raise ValueError("The 'dates' key is missing from the inputs.")

                start_date = format_date(inputs["dates"][0])
                end_date = format_date(inputs["dates"][1])
                input_date = parse_date_from_filename(im_meta)
                # if the image date is outside the specified date range, skip it
                if input_date < start_date or input_date > end_date:
                    continue
                meta_filepath = os.path.join(filepath_meta, im_meta)
                meta_info = read_metadata_file(meta_filepath)

                # Append meta info to the appropriate lists in the metadata dictionary
                metadata[satname]["filenames"].append(meta_info["filename"])
                metadata[satname]["acc_georef"].append(meta_info["acc_georef"])
                metadata[satname]["epsg"].append(meta_info["epsg"])
                metadata[satname]["dates"].append(
                    parse_date_from_filename(meta_info["filename"])
                )
                metadata[satname]["im_quality"].append(meta_info["im_quality"])
                if satname == "S1":
                    # appended unconditionally so it stays the same length as 'filenames'
                    metadata[satname]["band_order"].append(
                        get_band_order_from_meta(meta_info)
                    )
                # if the metadata file didn't contain im_height or im_width set this as an empty list
                if meta_info["im_height"] == -1 or meta_info["im_height"] == -1:
                    metadata[satname]["im_dimensions"].append([])
                else:
                    metadata[satname]["im_dimensions"].append(
                        [meta_info["im_height"], meta_info["im_width"]]
                    )

    # save a json file containing the metadata dict
    metadata_json = os.path.join(filepath, f"{inputs['sitename']}_metadata.json")
    SDS_preprocess.write_to_json(metadata_json, metadata)

    return metadata


###################################################################################################
# AUXILIARY FUNCTIONS
###################################################################################################


def get_downloaded_sar_polarizations(inputs: dict, metadata_s1: dict) -> dict:
    """
    Maps each already-downloaded Sentinel-1 date to the polarizations present for it.

    Prefers what is actually on disk - get_filenames returns only files that exist - and
    falls back to the 'band_order' recorded in the metadata when the files cannot be
    listed. Reading the disk means a scene whose VV was deleted by hand is correctly seen
    as incomplete.

    Args:
        inputs (dict): must contain 'filepath' and 'sitename'.
        metadata_s1 (dict): the 'S1' entry of the metadata dict, whose 'dates',
            'filenames' and 'band_order' lists are index-aligned.

    Returns:
        dict: {date: set of polarizations present}
    """
    try:
        filepath = SDS_tools.get_filepath(inputs, "S1")
    except Exception:
        filepath = None

    dates = metadata_s1.get("dates", [])
    filenames = metadata_s1.get("filenames", [])
    band_orders = metadata_s1.get("band_order", [])

    by_date = {}
    for index, date in enumerate(dates):
        present = set()
        if filepath and index < len(filenames):
            try:
                for file_path in SDS_tools.get_filenames(
                    filenames[index], filepath, "S1"
                ):
                    polar = SDS_tools.get_polarization_from_filename(file_path)
                    if polar:
                        present.add(polar)
            except (FileNotFoundError, IOError, IndexError):
                present = set()
        if not present and index < len(band_orders):
            # no readable files, trust what the metadata says was downloaded
            bands = band_orders[index]
            if isinstance(bands, str):
                bands = [bands]
            present = {str(b).strip().upper() for b in bands if str(b).strip()}
        by_date.setdefault(date, set()).update(present)

    return by_date


def get_incomplete_sar_dates(
    inputs: dict, metadata_s1: dict, polarizations: list
) -> set:
    """
    Dates of downloaded Sentinel-1 scenes that are missing a requested polarization.

    Those scenes have to be downloaded again on a resume, or a site first downloaded as
    VH-only would stay single-polarization forever and never reach the segmentation
    model, which needs both VV and VH.

    Args:
        inputs (dict): must contain 'filepath' and 'sitename'.
        metadata_s1 (dict): the 'S1' entry of the metadata dict.
        polarizations (list): the polarizations requested for this run.

    Returns:
        set: dates whose scene does not carry every requested polarization
    """
    requested = {str(polar).strip().upper() for polar in polarizations}
    return {
        date
        for date, present in get_downloaded_sar_polarizations(
            inputs, metadata_s1
        ).items()
        if not requested.issubset(present)
    }


def remove_existing_imagery(
    image_dict: dict, metadata: dict, sat_list: list[str], inputs: dict
) -> dict:
    """
    Removes existing imagery from the image dictionary based on the provided metadata.

    Args:
        image_dict (dict): A dictionary containing satellite imagery data.
            Each image_dict contains a list of images for each satellite.
            Each image in the list must contain the keys ['properties']['system:time_start']
            Each image is a dictionary containing the image properties, dates and metadata obtained from the server it was downloaded from.
            Example:
            {
                'L5': [{'type': 'Image',
                        'bands': [{'id': 'B1',
                        'data_type': {'type': 'PixelType',
                        'precision': 'int',
                        'min': 0,
                        'max': 65535},
                        'dimensions': [1830, 1830],
                        'crs': 'EPSG:32618',
                        'crs_transform': [60, 0, 499980, 0, -60, 4500000]},...},
                {...}, ...],
            }
        metadata (dict): A dictionary containing metadata information.
            Contains metadata for each satellite in the following format:
            {
                'L5':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'L7':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'L8':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'L9':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]},
                'S2':{'filenames':[], 'dates':[], 'epsg':[], 'acc_georef':[], 'im_quality':[], 'im_dimensions':[]}
            }
        sat_list (list[str]): A list of satellite names.
        inputs (dict): the run inputs, giving the requested Sentinel-1 polarizations and
            where the images live. Required: without it an S1 scene missing a
            polarization cannot be told apart from a complete one, and the site would
            silently stay single-polarization on every resume.

    Returns:
        dict: The updated image dictionary after removing existing imagery.
    """
    for satname in sat_list:
        if satname not in metadata:
            avail_date_list = [
                datetime.fromtimestamp(
                    image["properties"]["system:time_start"] / 1000, tz=pytz.utc
                ).replace(microsecond=0)
                for image in image_dict[satname]
            ]
            print(
                f"{satname}:There are {len(avail_date_list)} images available, 0 images already exist, {len(avail_date_list)} to download"
            )
        # If the satellite name exists in the dictionary of already downloaded images
        if satname in metadata and metadata[satname]["dates"]:
            avail_date_list = [
                datetime.fromtimestamp(
                    image["properties"]["system:time_start"] / 1000, tz=pytz.utc
                ).replace(microsecond=0)
                for image in image_dict[satname]
            ]
            if len(avail_date_list) == 0:
                print(
                    f'{satname}:There are {len(avail_date_list)} images available, {len(metadata[satname]["dates"])} images already exist, {len(avail_date_list)} to download'
                )
                continue
            downloaded_dates = metadata[satname]["dates"]
            if len(downloaded_dates) == 0:
                print(
                    f"{satname}:There are {len(avail_date_list)} images available, {len(downloaded_dates)} images already exist, {len(avail_date_list)} to download"
                )
                continue

            # A Sentinel-1 scene only counts as downloaded once every requested
            # polarization is there. Matching on the date alone would leave a site that
            # was first downloaded as VH-only stuck single-polarization on every resume,
            # and single-polarization scenes cannot be segmented by the model.
            if satname == "S1":
                polarizations = get_sentinel1_polarizations(inputs)
                incomplete_dates = get_incomplete_sar_dates(
                    inputs, metadata[satname], polarizations
                )
                if incomplete_dates:
                    downloaded_dates = [
                        date
                        for date in downloaded_dates
                        if date not in incomplete_dates
                    ]
                    print(
                        f"{satname}: {len(incomplete_dates)} existing images are missing "
                        f"one of {'+'.join(polarizations)} and will be downloaded again"
                    )

            # get the indices of the images that are not already downloaded
            idx_new = np.where(
                [not avail_date in downloaded_dates for avail_date in avail_date_list]
            )[0]
            image_dict[satname] = [image_dict[satname][index] for index in idx_new]
            print(
                f"{satname}:There are {len(avail_date_list)} images available, {len(downloaded_dates)} images already exist, {len(idx_new)} to download"
            )
    return image_dict


def get_s2cloudless(
    image_list: list,
    inputs: dict,
):
    """
    Match the list of Sentinel-2 (S2) images with the corresponding s2cloudless images.

    Parameters:
    image_list (List[Dict]): A list of dictionaries, each representing metadata for an S2 image.
    inputs (Dict[str, Union[str, List[str]]]): A dictionary containing:
        - 'dates': List of dates in 'YYYY-mm-dd' format as strings.
        - 'polygon': A list of coordinates defining the polygon of interest.
        - 'min_roi_coverage': Minimum required coverage of the region of interest by the image (default is 0.3).
    Returns:
    List[Union[Dict, List]]: A list where each element is either a dictionary containing metadata of a matched
    s2cloudless image or an empty list if no match is found.
    """
    # Convert string dates to datetime objects
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in inputs["dates"]]
    polygon = inputs["polygon"]
    collection = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    cloud_images_list = get_images_list_from_collection(
        collection,
        polygon,
        dates,
        min_roi_coverage=inputs.get("min_roi_coverage", 0.3),
    )
    # Extract image IDs from the s2cloudless collection
    cloud_indices = [image["properties"]["system:index"] for image in cloud_images_list]
    # match with S2 images
    matched_cloud_images = []
    for image_meta in image_list:
        index = image_meta["properties"]["system:index"]

        if index in cloud_indices:
            matched_index = cloud_indices.index(index)
            matched_cloud_images.append(cloud_images_list[matched_index])
        else:  # append an empty list if no match is found
            matched_cloud_images.append([])

    return matched_cloud_images


def remove_cloudy_images(im_list, satname, cloud_threshold=0.95, **kwargs):
    """
    Removes from the EE collection very cloudy images (>95% cloud cover)

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection
    satname:
        name of the satellite mission
    cloud_threshold: float
        percentage of cloud cover acceptable on the images

    Returns:
    -----------
    im_list_upt: list
        updated list of images
    """
    # Convert threshold to percentage
    cloud_threshold *= 100

    # Determine correct property name
    if satname in ["S1"]:
        return im_list  # No cloud cover property for S1
    if satname in ["L5", "L7", "L8", "L9"]:
        cloud_property = "CLOUD_COVER"
    elif satname in ["S2"]:
        cloud_property = "CLOUDY_PIXEL_PERCENTAGE"
    else:
        raise ValueError(f"Unknown satellite name: {satname}")

    # Filter the image list
    im_list_upt = [
        img
        for img in im_list
        if img["properties"].get(cloud_property, 0) <= cloud_threshold
    ]

    return im_list_upt


@retry
def adjust_polygon(polygon, proj, **kwargs):
    """
    Adjust polygon of ROI to fit exactly with the pixels of the underlying tile

    KV WRL 2022

    Arguments:
    -----------
    polygon: list
        polygon containing the lon/lat coordinates to be extracted,
        longitudes in the first column and latitudes in the second column,
        there are 5 pairs of lat/lon with the fifth point equal to the first point:
        ```
        polygon = [[[151.3, -33.7],[151.4, -33.7],[151.4, -33.8],[151.3, -33.8],
        [151.3, -33.7]]]
        ```
    proj: ee.Proj
        projection of the underlying tile

    Returns:
    -----------
    ee_region: ee
        updated list of images
    """
    # adjust polygon to match image coordinates so that there is no resampling
    polygon_ee = ee.Geometry.Polygon(polygon)
    # convert polygon to image coordinates
    polygon_coords = np.array(
        ee.List(polygon_ee.transform(proj, 1).coordinates().get(0)).getInfo()
    )
    # make it a rectangle
    xmin = np.min(polygon_coords[:, 0])
    ymin = np.min(polygon_coords[:, 1])
    xmax = np.max(polygon_coords[:, 0])
    ymax = np.max(polygon_coords[:, 1])
    # round to the closest pixels
    rect = [np.floor(xmin), np.floor(ymin), np.ceil(xmax), np.ceil(ymax)]
    # convert back to epsg 4326
    ee_region = ee.Geometry.Rectangle(rect, proj, True, False).transform("EPSG:4326")
    return ee_region


# decorator to try the download up to 3 times
@retry
def download_tif(
    image: ee.Image,
    polygon: list,
    bands: list[dict],
    filepath: str,
    satname: str,
    **kwargs,
) -> Union[str, Tuple[str, str]]:
    """
    Downloads a .TIF image from the ee server. The image is downloaded as a
    zip file then moved to the working directory, unzipped and stacked into a
    single .TIF file. Any QA band is saved separately.

    KV WRL 2018

    Arguments:
    -----------
    image: ee.Image
        Image object to be downloaded
    polygon: list
        polygon containing the lon/lat coordinates to be extracted
        longitudes in the first column and latitudes in the second column
    bands: list of dict
        list of bands to be downloaded
    filepath: str
        location where the temporary file should be saved
    satname: str
        name of the satellite missions ['L5','L7','L8','S2']
    Returns:
    -----------
    Downloads an image in a file named data.tif

    """
    # crop and download
    download_id = ee.data.getDownloadId(
        {
            "image": image,
            "region": polygon,
            "bands": bands,
            "filePerBand": True,
            "name": "image",
        }
    )
    response = requests.get(
        ee.data.makeDownloadUrl(download_id),
        timeout=(30, 30),  # 30 seconds to connect, 30 seconds to read
    )
    fp_zip = os.path.join(filepath, "temp.zip")
    with open(fp_zip, "wb") as fd:
        fd.write(response.content)
    # unzip the individual bands
    with zipfile.ZipFile(fp_zip) as local_zipfile:
        for fn in local_zipfile.namelist():
            local_zipfile.extract(fn, filepath)
        fn_all = [os.path.join(filepath, _) for _ in local_zipfile.namelist()]
    os.remove(fp_zip)
    # now process the individual bands:
    # - for Sentinel-1
    if satname in ["S1"]:
        # if there is only one band, it's either the SWIR1 or QA60
        if len(fn_all) == 1:
            # return the filename of the .tif
            return fn_all[0]
        # otherwise there are multiple multispectral bands so we have to merge them into one .tif
        else:
            # select all ms bands except the QA band (which is processed separately)
            fn_tifs = fn_all
            filename = "ms_bands.tif"
            # build a VRT and merge the bands (works the same with pan band)
            outds = gdal.BuildVRT(
                os.path.join(filepath, "temp.vrt"), fn_tifs, separate=True
            )
            outds = gdal.Translate(os.path.join(filepath, filename), outds)
            # remove temporary files
            os.remove(os.path.join(filepath, "temp.vrt"))
            for _ in fn_tifs:
                os.remove(_)
            if os.path.exists(os.path.join(filepath, filename + ".aux.xml")):
                os.remove(os.path.join(filepath, filename + ".aux.xml"))
            # return filename of the merge .tif file
            fn_image = os.path.join(filepath, filename)
            return fn_image

    # - for Landsat
    if satname in ["L5", "L7", "L8", "L9"]:
        # if there is only one band, it's the panchromatic
        if len(fn_all) == 1:
            # return the filename of the .tif
            return fn_all[0]
        # otherwise there are multiple multispectral bands so we have to merge them into one .tif
        else:
            # select all ms bands except the QA band (which is processed separately)
            fn_tifs = [_ for _ in fn_all if not "QA" in _]
            filename = "ms_bands.tif"
            # build a VRT and merge the bands (works the same with pan band)
            outds = gdal.BuildVRT(
                os.path.join(filepath, "temp.vrt"), fn_tifs, separate=True
            )
            outds = gdal.Translate(os.path.join(filepath, filename), outds)
            # remove temporary files
            os.remove(os.path.join(filepath, "temp.vrt"))
            for _ in fn_tifs:
                os.remove(_)
            if os.path.exists(os.path.join(filepath, filename + ".aux.xml")):
                os.remove(os.path.join(filepath, filename + ".aux.xml"))
            # return file names (ms and QA bands separately)
            fn_image = os.path.join(filepath, filename)
            fn_QA = [_ for _ in fn_all if "QA" in _][0]
            return fn_image, fn_QA

    # - for Sentinel-2
    if satname in ["S2"]:
        # if there is only one band, it's either the SWIR1 or QA60
        if len(fn_all) == 1:
            # return the filename of the .tif
            return fn_all[0]
        # otherwise there are multiple multispectral bands so we have to merge them into one .tif
        else:
            # select all ms bands except the QA band (which is processed separately)
            fn_tifs = fn_all
            filename = "ms_bands.tif"
            # build a VRT and merge the bands (works the same with pan band)
            outds = gdal.BuildVRT(
                os.path.join(filepath, "temp.vrt"), fn_tifs, separate=True
            )
            outds = gdal.Translate(os.path.join(filepath, filename), outds)
            # remove temporary files
            os.remove(os.path.join(filepath, "temp.vrt"))
            for _ in fn_tifs:
                os.remove(_)
            if os.path.exists(os.path.join(filepath, filename + ".aux.xml")):
                os.remove(os.path.join(filepath, filename + ".aux.xml"))
            # return filename of the merge .tif file
            fn_image = os.path.join(filepath, filename)
            return fn_image


def warp_image_to_target(
    fn_in, fn_out, fn_target, double_res=True, resampling_method="bilinear"
):
    """
    Resample an image on a new pixel grid based on a target image using gdal_warp.
    This is used to align the multispectral and panchromatic bands, as well as just downsample certain bands.

    KV WRL 2022

    Arguments:
    -----------
    fn_in: str
        filepath of the input image (points to .tif file)
    fn_out: str
        filepath of the output image (will be created)
    fn_target: str
        filepath of the target image
    double_res: boolean
        this function can be used to downsample images by settings the input and target
        filepaths to the same imageif the input and target images are the same and settings
        double_res = True to downsample by a factor of 2
    resampling_method: str
        method using to resample the image on the new pixel grid. See gdal_warp documentation
        for options (https://gdal.org/programs/gdalwarp.html)

    Returns:
    -----------
    Creates a new .tif file (fn_out)

    """
    # get output extent from target image
    im_target = gdal.Open(fn_target, gdal.GA_ReadOnly)
    georef_target = np.array(im_target.GetGeoTransform())
    xres = georef_target[1]
    yres = georef_target[5]
    if double_res:
        xres = int(georef_target[1] / 2)
        yres = int(georef_target[5] / 2)
    extent_pan = SDS_tools.get_image_bounds(fn_target)
    extent_coords = np.array(extent_pan.exterior.coords)
    xmin = np.min(extent_coords[:, 0])
    ymin = np.min(extent_coords[:, 1])
    xmax = np.max(extent_coords[:, 0])
    ymax = np.max(extent_coords[:, 1])

    # use gdal_warp to resample the input onto the target image pixel grid
    options = gdal.WarpOptions(
        xRes=xres,
        yRes=yres,
        outputBounds=[xmin, ymin, xmax, ymax],
        resampleAlg=resampling_method,
        targetAlignedPixels=False,
    )
    gdal.Warp(fn_out, fn_in, options=options)

    # check that both files have the same georef and size (important!)
    im_target = gdal.Open(fn_target, gdal.GA_ReadOnly)
    im_out = gdal.Open(fn_out, gdal.GA_ReadOnly)
    georef_target = np.array(im_target.GetGeoTransform())
    georef_out = np.array(im_out.GetGeoTransform())
    size_target = np.array([im_target.RasterXSize, im_target.RasterYSize])
    size_out = np.array([im_out.RasterXSize, im_out.RasterYSize])
    if double_res:
        size_target = size_target * 2
    if np.any(np.nonzero(georef_target[[0, 3]] - georef_out[[0, 3]])):
        raise Exception("Georef of pan and ms bands do not match for image %s" % fn_out)
    if np.any(np.nonzero(size_target - size_out)):
        raise Exception("Size of pan and ms bands do not match for image %s" % fn_out)


###################################################################################################
# Sentinel-2 functions
###################################################################################################


def filter_S2_collection(im_list):
    """
    Removes duplicates from the EE collection of Sentinel-2 images (many duplicates)
    Finds the images that were acquired at the same time but have different utm zones.

    KV WRL 2018

    Arguments:
    -----------
    im_list: list
        list of images in the collection

    Returns:
    -----------
    im_list_flt: list
        filtered list of images
    """
    if not im_list:
        return im_list

    # get datetimes
    timestamps = [
        datetime.fromtimestamp(_["properties"]["system:time_start"] / 1000, tz=pytz.utc)
        for _ in im_list
    ]
    # get utm zone projections
    utm_zones = np.array([int(_["bands"][0]["crs"][5:]) for _ in im_list])
    if len(np.unique(utm_zones)) == 1:
        return im_list
    else:
        idx_max = np.argmax([np.sum(utm_zones == _) for _ in np.unique(utm_zones)])
        utm_zone_selected = np.unique(utm_zones)[idx_max]
        # find the images that were acquired at the same time but have different utm zones
        idx_all = np.arange(0, len(im_list), 1)
        idx_covered = np.ones(len(im_list)).astype(bool)
        idx_delete = []
        i = 0
        while 1:
            same_time = (
                np.abs([(timestamps[i] - _).total_seconds() for _ in timestamps])
                < 60 * 60 * 24
            )
            idx_same_time = np.where(same_time)[0]
            same_utm = utm_zones == utm_zone_selected
            # get indices that have the same time (less than 24h apart) but not the same utm zone
            idx_temp = np.where(
                [same_time[j] == True and same_utm[j] == False for j in idx_all]
            )[0]
            idx_keep = idx_same_time[[_ not in idx_temp for _ in idx_same_time]]
            # if more than 2 images with same date and same utm, drop the last ones
            if len(idx_keep) > 2:
                idx_temp = np.append(idx_temp, idx_keep[-(len(idx_keep) - 2) :])
            for j in idx_temp:
                idx_delete.append(j)
            idx_covered[idx_same_time] = False
            if np.any(idx_covered):
                i = np.where(idx_covered)[0][0]
            else:
                break
        # update the collection by deleting all those images that have same timestamp
        # and different utm projection
        im_list_flt = [x for k, x in enumerate(im_list) if k not in idx_delete]

    return im_list_flt
