"""
This module contains functions to analyze the 2D shorelines along shore-normal
transects
    
Author: Kilian Vos, Water Research Laboratory, University of New South Wales
"""

# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import pdb

# other modules
import skimage.transform as transform
from pylab import ginput
from scipy import stats

# CoastSat modules
from coastsat import SDS_tools

from tqdm.auto import tqdm

# Global variables
DAYS_IN_YEAR = 365.2425
SECONDS_IN_DAY = 24*3600

###################################################################################################
# DRAW/LOAD TRANSECTS
###################################################################################################


def create_transect(origin, orientation, length):
    """
    Create a transect given an origin, orientation and length.
    Points are spaced at 1m intervals.

    KV WRL 2018

    Arguments:
    -----------
    origin: np.array
        contains the X and Y coordinates of the origin of the transect
    orientation: int
        angle of the transect (anti-clockwise from North) in degrees
    length: int
        length of the transect in metres

    Returns:
    -----------
    transect: np.array
        contains the X and Y coordinates of the transect

    """

    # origin of the transect
    x0 = origin[0]
    y0 = origin[1]
    # orientation of the transect
    phi = (90 - orientation) * np.pi / 180
    # create a vector with points at 1 m intervals
    x = np.linspace(0, length, length + 1)
    y = np.zeros(len(x))
    coords = np.zeros((len(x), 2))
    coords[:, 0] = x
    coords[:, 1] = y
    # translate and rotate the vector using the origin and orientation
    tf = transform.EuclideanTransform(rotation=phi, translation=(x0, y0))
    transect = tf(coords)

    return transect


def draw_transects(output, settings):
    """
    Draw shore-normal transects interactively on top of the mapped shorelines

    KV WRL 2018

    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    settings: dict with the following keys
        'inputs': dict
            input parameters (sitename, filepath, polygon, dates, sat_list)

    Returns:
    -----------
    transects: dict
        contains the X and Y coordinates of all the transects drawn.
        Also saves the coordinates as a .geojson as well as a .jpg figure
        showing the location of the transects.
    """

    sitename = settings["inputs"]["sitename"]
    filepath = os.path.join(settings["inputs"]["filepath"], sitename)

    # plot the mapped shorelines
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.axis("equal")
    ax1.set_xlabel("Eastings [m]")
    ax1.set_ylabel("Northings [m]")
    ax1.grid(linestyle=":", color="0.5")
    for i in range(len(output["shorelines"])):
        sl = output["shorelines"][i]
        date = output["dates"][i]
        ax1.plot(sl[:, 0], sl[:, 1], ".", markersize=3, label=date.strftime("%d-%m-%Y"))
    #    ax1.legend()
    fig1.set_tight_layout(True)
    mng = plt.get_current_fig_manager()
    ax1.set_title(
        "Click two points to define each transect (first point is the "
        + "origin of the transect and is landwards, second point seawards).\n"
        + "When all transects have been defined, click on <ENTER>",
        fontsize=16,
    )

    # initialise transects dict
    transects = dict([])
    counter = 0
    # loop until user breaks it by click <enter>
    while 1:
        # let user click two points
        pts = ginput(n=2, timeout=-1)
        if len(pts) > 0:
            origin = pts[0]
        # if user presses <enter>, no points are selected
        else:
            # save figure as .jpg
            fig1.gca().set_title("Transect locations", fontsize=16)
            fig1.savefig(
                os.path.join(
                    filepath, "jpg_files", sitename + "_transect_locations.jpg"
                ),
                dpi=200,
            )
            plt.title(
                "Transect coordinates saved as " + sitename + "_transects.geojson"
            )
            plt.draw()
            # wait 2 seconds for user to visualise the transects that are saved
            ginput(n=1, timeout=2, show_clicks=True)
            plt.close(fig1)
            # break the loop
            break

        # add selectect points to the transect dict
        counter = counter + 1
        transect = np.array([pts[0], pts[1]])

        # alternative of making the transect the origin, orientation and length
        #        temp = np.array(pts[1]) - np.array(origin)
        #        phi = np.arctan2(temp[1], temp[0])
        #        orientation = -(phi*180/np.pi - 90)
        #        length = np.linalg.norm(temp)
        #        transect = create_transect(origin, orientation, length)

        transects[str(counter)] = transect

        # plot the transects on the figure
        ax1.plot(transect[:, 0], transect[:, 1], "b-", lw=2.5)
        ax1.plot(transect[0, 0], transect[0, 1], "rx", markersize=10)
        ax1.text(
            transect[-1, 0],
            transect[-1, 1],
            str(counter),
            size=16,
            bbox=dict(boxstyle="square", ec="k", fc="w"),
        )
        plt.draw()

    # save transects.geojson
    gdf = SDS_tools.transects_to_gdf(transects)
    # set projection
    gdf.crs = f"epsg:{settings['output_epsg']}"
    # save as geojson
    gdf.to_file(
        os.path.join(filepath, sitename + "_transects.geojson"),
        driver="GeoJSON",
        encoding="utf-8",
    )
    # print the location of the files
    print("Transect locations saved in " + filepath)

    return transects


###################################################################################################
# COMPUTE INTERSECTIONS
###################################################################################################


def compute_intersection(output, transects, settings):
    """
    Computes the intersection between the 2D shorelines and the shore-normal.
    transects. It returns time-series of cross-shore distance along each transect.

    KV WRL 2018
    Arguments:
    -----------
    output: dict
        contains the extracted shorelines and corresponding metadata
    transects: dict
        contains the X and Y coordinates of each transect
    settings: dict with the following keys
        'along_dist': int
            alongshore distance considered caluclate the intersection

    Returns:
    -----------
    cross_dist: dict
        time-series of cross-shore distance along each of the transects.
        Not tidally corrected.
    """

    # loop through shorelines and compute the median intersection
    intersections = np.zeros((len(output["shorelines"]), len(transects)))
    for i in range(len(output["shorelines"])):
        sl = output["shorelines"][i]

        for j, key in enumerate(list(transects.keys())):
            # compute rotation matrix
            X0 = transects[key][0, 0]
            Y0 = transects[key][0, 1]
            temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0, Y0])
            p2 = transects[key][-1, :]
            # 2-D cross product written out. np.cross on 2-vectors is deprecated in
            # NumPy 2.0 and slated for removal; for a=(ax,ay) and b=(bx,by) the only
            # non-zero component is ax*by - ay*bx, which is all this distance needs.
            v = p2 - p1
            w = sl - p1
            d_line = np.abs((v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v))
            # calculate the distance between shoreline points and the origin of the transect
            # vectorized: `w` is already sl - p1, so this is a single row-wise norm
            # instead of one np.linalg.norm call per shoreline point
            d_origin = np.linalg.norm(w, axis=1)
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km
            idx_dist = np.logical_and(
                d_line <= settings["along_dist"], d_origin <= 1000
            )
            # find the shoreline points that are in the direction of the transect (within 90 degrees)
            temp_sl = sl - np.array(transects[key][0, :])
            phi_sl = np.array(
                [np.arctan2(temp_sl[k, 1], temp_sl[k, 0]) for k in range(len(temp_sl))]
            )
            diff_angle = phi - phi_sl
            idx_angle = np.abs(diff_angle) < np.pi / 2
            # combine the transects that are close in distance and close in orientation
            idx_close = np.where(np.logical_and(idx_dist, idx_angle))[0]

            # in case there are no shoreline points close to the transect
            if len(idx_close) == 0:
                intersections[i, j] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
                    np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
                )
                xy_rot = np.matmul(Mrot, xy_close)
                # compute the median of the intersections along the transect
                intersections[i, j] = np.nanmedian(xy_rot[0, :])

    # fill the a dictionnary
    cross_dist = dict([])
    for j, key in enumerate(list(transects.keys())):
        cross_dist[key] = intersections[:, j]

    return cross_dist


def compute_intersection_QC(output, transects, settings, use_progress_bar: bool = True):
    """
    More advanced function to omputes the intersection between the 2D mapped shorelines
    and the transects. Produces more quality-controlled time-series of shoreline change.

    Arguments:
    -----------
        output: dict
            contains the extracted shorelines and corresponding dates.
        transects: dict
            contains the X and Y coordinates of the transects (first and last point needed for each
            transect).
        settings: dict
                'along_dist': int (in metres)
                    alongshore distance to caluclate the intersection (median of points
                    within this distance).
                'min_points': int
                    minimum number of shoreline points to calculate an intersections
                'max_std': int (in metres)
                    maximum std for the shoreline points when calculating the median,
                    if above this value then NaN is returned for the intersection
                'max_range': int (in metres)
                    maximum range  for the shoreline points when calculating the median,
                    if above this value then NaN is returned for the intersection
                'min_chainage': int (in metres)
                    furthest landward of the transect origin that an intersection is
                    accepted, beyond this point a NaN is returned
                'multiple_inter': mode for removing outliers ('auto', 'nan', 'max')
                'prc_multiple': percentage to use in 'auto' mode to switch from 'nan' to 'max'

        use_progress_bar(bool,optional). Defaults to True. If true uses tqdm to display the progress for iterating through transects.
        False, means no progress bar is displayed

    Returns:
    -----------
        cross_dist: dict
            time-series of cross-shore distance along each of the transects. These are not tidally
            corrected.

    """

    # initialise dictionary with intersections for each transect
    cross_dist = dict([])

    shorelines = output["shorelines"]
    along_dist = settings["along_dist"]

    # loop through each transect
    transect_keys = transects.keys()
    if use_progress_bar:
        transect_keys = tqdm(
            transect_keys, desc="Computing transect shoreline intersections"
        )

    for key in transect_keys:
        # initialise variables
        std_intersect = np.zeros(len(shorelines))
        med_intersect = np.zeros(len(shorelines))
        max_intersect = np.zeros(len(shorelines))
        min_intersect = np.zeros(len(shorelines))
        n_intersect = np.zeros(len(shorelines))

        # loop through each shoreline
        for i in range(len(shorelines)):
            sl = shorelines[i]

            # in case there are no shoreline points
            if len(sl) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
                continue

            # compute rotation matrix
            X0 = transects[key][0, 0]
            Y0 = transects[key][0, 1]
            temp = np.array(transects[key][-1, :]) - np.array(transects[key][0, :])
            phi = np.arctan2(temp[1], temp[0])
            Mrot = np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

            # calculate point to line distance between shoreline points and the transect
            p1 = np.array([X0, Y0])
            p2 = transects[key][-1, :]
            # 2-D cross product written out. np.cross on 2-vectors is deprecated in
            # NumPy 2.0 and slated for removal; for a=(ax,ay) and b=(bx,by) the only
            # non-zero component is ax*by - ay*bx, which is all this distance needs.
            v = p2 - p1
            w = sl - p1
            d_line = np.abs((v[0] * w[:, 1] - v[1] * w[:, 0]) / np.linalg.norm(v))
            # calculate the distance between shoreline points and the origin of the transect
            # vectorized: `w` is already sl - p1, so this is a single row-wise norm
            # instead of one np.linalg.norm call per shoreline point
            d_origin = np.linalg.norm(w, axis=1)
            # find the shoreline points that are close to the transects and to the origin
            # the distance to the origin is hard-coded here to 1 km
            idx_dist = np.logical_and(d_line <= along_dist, d_origin <= 1000)
            idx_close = np.where(idx_dist)[0]

            # in case there are no shoreline points close to the transect
            if len(idx_close) == 0:
                std_intersect[i] = np.nan
                med_intersect[i] = np.nan
                max_intersect[i] = np.nan
                min_intersect[i] = np.nan
                n_intersect[i] = np.nan
            else:
                # change of base to shore-normal coordinate system
                xy_close = np.array([sl[idx_close, 0], sl[idx_close, 1]]) - np.tile(
                    np.array([[X0], [Y0]]), (1, len(sl[idx_close]))
                )
                xy_rot = np.matmul(Mrot, xy_close)
                # remove points that are too far landwards relative to the transect origin (i.e., negative chainage)
                xy_rot[0, xy_rot[0, :] < settings["min_chainage"]] = np.nan

                # compute std, median, max, min of the intersections
                if not np.all(np.isnan(xy_rot[0, :])):
                    std_intersect[i] = np.nanstd(xy_rot[0, :])
                    med_intersect[i] = np.nanmedian(xy_rot[0, :])
                    max_intersect[i] = np.nanmax(xy_rot[0, :])
                    min_intersect[i] = np.nanmin(xy_rot[0, :])
                    n_intersect[i] = np.sum(~np.isnan(xy_rot[0, :])) # count only non-nan values
                else:
                    std_intersect[i] = np.nan
                    med_intersect[i] = np.nan
                    max_intersect[i] = np.nan
                    min_intersect[i] = np.nan
                    n_intersect[i] = 0

        # quality control the intersections using dispersion metrics (std and range)
        condition1 = std_intersect <= settings["max_std"]
        condition2 = (max_intersect - min_intersect) <= settings["max_range"]
        condition3 = n_intersect >= settings["min_points"]
        idx_good = np.logical_and(np.logical_and(condition1, condition2), condition3)

        # save copy for QA plot later
        # med_intersect_temp = med_intersect.copy()
        # max_intersect_temp = max_intersect.copy()

        # decide what to do with the intersections with high dispersion
        if settings["multiple_inter"] == "auto":
            # compute the percentage of data points where the std is larger than the user-defined max
            prc_over = np.sum(std_intersect > settings["max_std"]) / len(std_intersect)
            # if more than a certain percentage is above, use the maximum intersection
            
            prc_multiple = settings.get("prc_multiple")
            if prc_multiple is None:
                prc_multiple = settings.get("auto_prc")
                if prc_multiple is None:
                    raise KeyError("Neither 'prc_multiple' nor 'auto_prc' exist in the settings.")
            
            if prc_over > prc_multiple:
                med_intersect[~idx_good] = max_intersect[~idx_good]
                med_intersect[~condition3] = np.nan
            # otherwise put a nan
            else:
                med_intersect[~idx_good] = np.nan

        elif settings["multiple_inter"] == "max":
            med_intersect[~idx_good] = max_intersect[~idx_good]
            med_intersect[~condition3] = np.nan
            prc_over = 0

        elif settings["multiple_inter"] == "nan":
            med_intersect[~idx_good] = np.nan
            prc_over = 0

        else:
            raise Exception(
                "the multiple_inter parameter can only be: nan, max or auto"
            )

        # store in dict
        cross_dist[key] = med_intersect

        # plot for troubleshooting
        # if settings['plot_fig']:
        #     fig,ax=plt.subplots(1,1,figsize=[12,4])
        #     fig.set_tight_layout(True)
        #     ax.grid(linestyle=':', color='0.5')
        #     ax.plot(output['dates'], med_intersect_temp, 'o-', c='0.6', ms=3, label='median')
        #     ax.plot(output['dates'], max_intersect_temp, 'o-', c='g', ms=3, label='max')
        #     ax.plot(output['dates'], cross_dist[key], 'o', c='r', ms=3, label='final')
        #     ax.set_title('%s - %.2f'%(key, prc_over))
        #     ax.legend(loc=3)
        #     print('%s  - removed %d'%(key, sum(np.isnan(med_intersect))))

    return cross_dist


###################################################################################################
# DESPIKING/OUTLIER REMOVAL
###################################################################################################


def threshold_filter_applies(output):
    """
    Per shoreline, whether settings['otsu_threshold'] can be compared to its threshold.

    output['MNDWI_threshold'] does not hold the same quantity for every shoreline. Only
    the optical path stores an MNDWI value there; SAR stores an Otsu threshold in dB, or
    NaN when the segmentation model mapped the shoreline and there is no threshold at
    all. Comparing those against an MNDWI range silently drops them - NaN fails every
    comparison, and a dB value near -20 falls outside any sensible MNDWI range - so the
    shorelines this returns False for skip the filter instead of being discarded by it.

    Falls back to the satellite name for outputs written before 'segmentation_method'
    existed, and to True when neither key is present, which keeps the original behaviour
    for purely optical outputs.

    Arguments:
    -----------
    output: dict
        merged output dict, as returned by SDS_tools.merge_output

    Returns:
    -----------
    np.ndarray: boolean, one entry per shoreline

    """
    thresholds = np.array(output["MNDWI_threshold"], dtype=float)
    n_shorelines = len(thresholds)

    if "segmentation_method" in output:
        comparable = np.array(
            [
                method == SDS_tools.SEGMENTATION_MNDWI
                for method in output["segmentation_method"]
            ]
        )
    elif "satname" in output:
        # older outputs: every SAR shoreline was thresholded in dB
        comparable = np.array([satname != "S1" for satname in output["satname"]])
    else:
        comparable = np.ones(n_shorelines, dtype=bool)

    # a NaN threshold can never satisfy a range, so it must not be filtered on
    return comparable & np.isfinite(thresholds)


# Water fractions closer to all-land or all-water than this mean the segmentation
# found (almost) no interface inside the reference buffer. Deliberately loose so it
# only drops degenerate scenes; tighten it per site via
# settings['sar_water_fraction_range'], or set that to None to disable the check.
DEFAULT_SAR_WATER_FRACTION_RANGE = (0.01, 0.99)


def sar_qc_keep(output, settings):
    """
    Per shoreline, whether it passes the SAR-specific QC. Non-SAR rows always pass.

    SAR shorelines skip the MNDWI threshold filter (see threshold_filter_applies), so
    they get their own checks here:

    - Water fraction: output['water_fraction'] holds the fraction of the reference
      shoreline buffer segmented as water. A value outside
      settings['sar_water_fraction_range'] (default DEFAULT_SAR_WATER_FRACTION_RANGE)
      means the scene segmented as (almost) all land or all water - there was no
      interface to contour, so the mapped shoreline is dropped. Rows without the
      column (outputs written before it existed) and NaN fractions are kept.
    - Otsu dB threshold: when settings['sar_otsu_threshold'] is set to
      (min_dB, max_dB), 'sar_otsu' rows whose threshold falls outside it are dropped,
      the SAR counterpart of settings['otsu_threshold']. Off by default, because a
      sensible dB range is site- and sea-state-dependent.

    Model-segmented rows have no threshold at all, so the water fraction is the only
    check that can judge them.

    Arguments:
    -----------
    output: dict
        merged output dict, as returned by SDS_tools.merge_output
    settings: dict
        may contain 'sar_water_fraction_range' and 'sar_otsu_threshold'

    Returns:
    -----------
    np.ndarray: boolean, one entry per shoreline, False where the row fails QC

    """
    n_shorelines = len(output["dates"])
    keep = np.ones(n_shorelines, dtype=bool)

    methods = output.get("segmentation_method")
    if methods is not None:
        is_sar = np.array(
            [
                method
                in (SDS_tools.SEGMENTATION_SAR_OTSU, SDS_tools.SEGMENTATION_SAR_MODEL)
                for method in methods
            ]
        )
        is_otsu = np.array(
            [method == SDS_tools.SEGMENTATION_SAR_OTSU for method in methods]
        )
    elif "satname" in output:
        # older outputs: every SAR shoreline was thresholded in dB
        is_sar = np.array([satname == "S1" for satname in output["satname"]])
        is_otsu = is_sar
    else:
        return keep

    fraction_range = settings.get(
        "sar_water_fraction_range", DEFAULT_SAR_WATER_FRACTION_RANGE
    )
    if fraction_range is not None and "water_fraction" in output:
        fractions = np.array(output["water_fraction"], dtype=float)
        judged = is_sar & np.isfinite(fractions)
        keep[
            judged
            & ((fractions < fraction_range[0]) | (fractions > fraction_range[1]))
        ] = False

    db_range = settings.get("sar_otsu_threshold")
    if db_range is not None and not np.isnan(db_range[0]):
        thresholds = np.array(output["MNDWI_threshold"], dtype=float)
        judged = is_otsu & np.isfinite(thresholds)
        keep[
            judged & ((thresholds < db_range[0]) | (thresholds > db_range[1]))
        ] = False

    return keep


def reject_outliers(cross_distance, output, settings):
    """

    Arguments:
    -----------
        cross_distance: dict
            time-series of shoreline change
        output: dict
            mapped shorelines with metadata
        settings: dict
                'max_cross_change': int (in metres)
                    maximum cross-shore change observable between consecutive timesteps
                'otsu_threshold': tuple (min_t, max_t)
                    min and max intensity threshold use for contouring the shoreline.
                    Only applied to shorelines whose threshold is an MNDWI value; SAR
                    shorelines skip it rather than being dropped by it, see
                    threshold_filter_applies.
                'sar_water_fraction_range': tuple, optional
                    (min, max) acceptable water fraction in the reference buffer for
                    SAR shorelines, default DEFAULT_SAR_WATER_FRACTION_RANGE; None
                    disables the check (see sar_qc_keep)
                'sar_otsu_threshold': tuple, optional
                    (min_dB, max_dB) range for 'sar_otsu' shorelines, the SAR
                    counterpart of 'otsu_threshold'; off when not set

    Returns:
    -----------
        chain_dict: dict
            contains the updated time-series of cross-shore distance with the corresponding dates

    """

    chain_dict = dict([])

    # which shorelines the otsu_threshold range can actually judge
    applies = threshold_filter_applies(output)
    # SAR shorelines skip that range, so they get their own QC (water fraction,
    # optional dB threshold range); True for every non-SAR shoreline
    sar_keep = sar_qc_keep(output, settings)

    for i, key in enumerate(list(cross_distance.keys())):
        chainage = cross_distance[key].copy()
        if sum(np.isnan(chainage)) == len(chainage):
            continue

        # 1. Remove nans and negative chainages
        idx_nonan = np.where(~np.isnan(chainage))[0]
        chainage1 = [chainage[k] for k in idx_nonan]
        dates1 = [output["dates"][k] for k in idx_nonan]
        #        satnames1 = [output['satname'][k] for k in idx_nonan]

        # 2. Threshold-based QC: the MNDWI range for optical shorelines, the
        #    SAR-specific checks for SAR ones
        keep = sar_keep[idx_nonan].copy()
        if not np.isnan(settings["otsu_threshold"][0]):
            threshold1 = np.array(
                [output["MNDWI_threshold"][k] for k in idx_nonan], dtype=float
            )
            # shorelines whose threshold is not an MNDWI value are kept as they are:
            # the range cannot judge them, and comparing anyway would drop every one
            filterable = applies[idx_nonan]
            keep &= ~filterable | (
                (threshold1 <= settings["otsu_threshold"][1])
                & (threshold1 >= settings["otsu_threshold"][0])
            )
        idx_thres = np.where(keep)[0]
        chainage2 = [chainage1[k] for k in idx_thres]
        dates2 = [dates1[k] for k in idx_thres]
        # the minimum-points guard is tied to the MNDWI range filter, as it always
        # was; the SAR QC must not start dropping whole transects of short series
        if not np.isnan(settings["otsu_threshold"][0]) and len(chainage2) < 30:
            continue

        # 3. Remove outliers based on despiking [iterative method]
        chainage3, dates3 = identify_outliers(
            chainage2, dates2, settings["max_cross_change"]
        )

        # fill with nans the indices to be removed from cross_distance
        idx_kept = []
        for date in output["dates"]:
            idx_kept.append(date in dates3)
        chainage[~np.array(idx_kept)] = np.nan
        # store in chain_dict
        chain_dict[key] = chainage

        # figure for QA
        if settings["plot_fig"]:
            fig, ax = plt.subplots(2, 1, figsize=[12, 6], sharex=True)
            fig.set_tight_layout(True)
            ax[0].grid(linestyle=":", color="0.5")
            ax[0].set(
                ylabel="distance [m]",
                title="Transect %s original time-series - %d points"
                % (key, len(chainage1)),
            )
            mean_cross_dist = np.nanmedian(chainage3)
            # plot the data points
            ax[0].plot(dates1, chainage1 - mean_cross_dist, "C0-")
            ax[0].plot(
                dates1,
                chainage1 - mean_cross_dist,
                "C2o",
                ms=4,
                mec="k",
                mew=0.7,
                label="otsu",
            )
            # plot the indices removed because of the threshold
            ax[0].plot(
                dates2,
                chainage2 - mean_cross_dist,
                "C3o",
                ms=4,
                mec="k",
                mew=0.7,
                label="spike",
            )
            ax[0].legend(ncol=2, loc="upper right")
            # plot the final time-series
            ax[0].plot(
                dates3, chainage3 - mean_cross_dist, "C0o", ms=4, mfc="w", mec="C0"
            )
            ax[1].grid(linestyle=":", color="0.5")
            ax[1].plot(
                dates3, chainage3 - mean_cross_dist, "C0-o", ms=4, mfc="w", mec="C0"
            )
            ax[1].set(
                ylabel="distance [m]",
                title="Post-processed time-series - %d points" % (len(chainage3)),
            )
            print("%s  - outliers removed: %d" % (key, len(dates1) - len(dates3)))

    return chain_dict


def identify_outliers(chainage, dates, cross_change):
    """
    Remove outliers based on despiking [iterative method]

    Arguments:
    -----------
    chainage: list
        time-series of shoreline change
    dates: list of datetimes
        correspondings dates
    cross_change: float
        threshold distance to identify a point as an outlier

    Returns:
    -----------
    chainage_temp: list
        time-series of shoreline change without outliers
    dates_temp: list of datetimes
        dates without outliers

    """

    # make a copy of the inputs
    chainage_temp = chainage.copy()
    dates_temp = dates.copy()

    # loop through the time-series always starting from the start
    # when an outlier is found, remove it and restart
    # repeat until no more outliers are found in the time-series
    k = 0
    while k < len(chainage_temp):
        for k in range(len(chainage_temp)):
            # check if the first point is an outlier
            if k == 0:
                # difference between 1st and 2nd point in the time-series
                diff = chainage_temp[k] - chainage_temp[k + 1]
                if np.abs(diff) > cross_change:
                    chainage_temp.pop(k)
                    dates_temp.pop(k)
                    break

            # check if the last point is an outlier
            elif k == len(chainage_temp) - 1:
                # difference between last and before last point in the time-series
                diff = chainage_temp[k] - chainage_temp[k - 1]
                if np.abs(diff) > cross_change:
                    chainage_temp.pop(k)
                    dates_temp.pop(k)
                    break

            # check if a point is an isolated outlier or in a group of 2 consecutive outliers
            else:
                # calculate the difference with the data point before and after
                diff_m1 = chainage_temp[k] - chainage_temp[k - 1]
                diff_p1 = chainage_temp[k] - chainage_temp[k + 1]
                # remove point if isolated outlier, distant from both neighbours
                condition1 = np.abs(diff_m1) > cross_change
                condition2 = np.abs(diff_p1) > cross_change
                # check that distance from neighbours has the same sign
                condition3 = np.sign(diff_p1) == np.sign(diff_m1)
                if np.logical_and(np.logical_and(condition1, condition2), condition3):
                    chainage_temp.pop(k)
                    dates_temp.pop(k)
                    break

                # check for 2 consecutive outliers in the time-series
                if k >= 2 and k < len(chainage_temp) - 2:
                    # calculate difference with the data around the neighbours of the point
                    diff_m2 = chainage_temp[k - 1] - chainage_temp[k - 2]
                    diff_p2 = chainage_temp[k + 1] - chainage_temp[k + 2]
                    # remove if there are 2 consecutive outliers (see conditions below)
                    condition4 = np.abs(diff_m2) > cross_change
                    condition5 = np.abs(diff_p2) > cross_change
                    condition6 = np.sign(diff_m1) == np.sign(diff_p2)
                    condition7 = np.sign(diff_p1) == np.sign(diff_m2)
                    # check for both combinations (1,5,6 and ,2,4,7)
                    if np.logical_and(
                        np.logical_and(condition1, condition5), condition6
                    ):
                        chainage_temp.pop(k)
                        dates_temp.pop(k)
                        break
                    elif np.logical_and(
                        np.logical_and(condition2, condition4), condition7
                    ):
                        chainage_temp.pop(k)
                        dates_temp.pop(k)
                        break

                    # also look for clusters of 3 outliers
                    else:
                        # increase the distance to make sure these are really outliers
                        condition4b = np.abs(diff_m2) > 1.5 * cross_change
                        condition5b = np.abs(diff_p2) > 1.5 * cross_change
                        condition8 = np.sign(diff_m2) == np.sign(diff_p2)
                        # if point is close to immediate neighbours but
                        # the neighbours are far from their neighbours, point is an outlier
                        if np.logical_and(
                            np.logical_and(
                                np.logical_and(condition4b, condition5b),
                                np.logical_and(~condition1, ~condition2),
                            ),
                            condition8,
                        ):
                            print("*", end="")
                            chainage_temp.pop(k)
                            dates_temp.pop(k)
                            break

        # if one full loop is completed (went through all the time-series without removing outlier)
        # then increment k to get out of the loop
        k = k + 1

    # return the time-series where the outliers have been removed
    return chainage_temp, dates_temp


###################################################################################################
# SEASONAL/MONTHLY AVERAGING
###################################################################################################


def seasonal_average(dates, chainages):
    # define the 4 seasons
    months = ["-%02d" % _ for _ in np.arange(1, 13)]
    seasons = np.array([1, 4, 7, 10])
    season_labels = ["DJF", "MAM", "JJA", "SON"]
    # put time-series into a pd.dataframe (easier to process)
    df = pd.DataFrame({"dates": dates, "chainage": chainages})
    df.set_index("dates", inplace=True)
    # initialise variables for seasonal averages
    dict_seasonal = dict([])
    for k, j in enumerate(seasons):
        dict_seasonal[season_labels[k]] = {"dates": [], "chainages": []}
    dates_seasonal = []
    chainage_seasonal = []
    season_ts = []
    for year in np.unique(df.index.year):
        # 4 seasons: DJF, MMA, JJA, SON
        for k, j in enumerate(seasons):
            # middle date
            date_seas = pytz.utc.localize(datetime(year, j, 1))
            # if j == 1: date_seas = pytz.utc.localize(datetime(year-1,12,31))
            # for the first season, take the December data from the year before
            if j == 1:
                chain_seas = np.array(
                    df[
                        str(year - 1)
                        + months[(j - 1) - 1] : str(year)
                        + months[(j - 1) + 1]
                    ]["chainage"]
                )
            else:
                chain_seas = np.array(
                    df[
                        str(year)
                        + months[(j - 1) - 1] : str(year)
                        + months[(j - 1) + 1]
                    ]["chainage"]
                )
            if len(chain_seas) == 0:
                continue
            else:
                dict_seasonal[season_labels[k]]["dates"].append(date_seas)
                dict_seasonal[season_labels[k]]["chainages"].append(np.mean(chain_seas))
                dates_seasonal.append(date_seas)
                chainage_seasonal.append(np.mean(chain_seas))
                season_ts.append(j)
    # convert chainages to np.array (easier to manipulate than a list)
    for seas in dict_seasonal.keys():
        dict_seasonal[seas]["chainages"] = np.array(dict_seasonal[seas]["chainages"])

    return (
        dict_seasonal,
        dates_seasonal,
        np.array(chainage_seasonal),
        np.array(season_ts),
    )


def monthly_average(dates, chainages):
    # define the 12 months
    months = ["-%02d" % _ for _ in np.arange(1, 13)]
    seasons = np.arange(1, 13)
    season_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    # put time-series into a pd.dataframe (easier to process)
    df = pd.DataFrame({"dates": dates, "chainage": chainages})
    df.set_index("dates", inplace=True)
    # initialise variables for seasonal averages
    dict_seasonal = dict([])
    for k, j in enumerate(seasons):
        dict_seasonal[season_labels[k]] = {"dates": [], "chainages": []}
    dates_seasonal = []
    chainage_seasonal = []
    season_ts = []
    for year in np.unique(df.index.year):
        # 4 seasons: DJF, MMA, JJA, SON
        for k, j in enumerate(seasons):
            # middle date
            date_seas = pytz.utc.localize(datetime(year, j, 15))
            if date_seas > dates[-1] - timedelta(days=30):
                break
            try:
                chain_seas = np.array(df[str(year) + months[k]]["chainage"])
            except:
                continue
            if len(chain_seas) == 0:
                continue
            else:
                dict_seasonal[season_labels[k]]["dates"].append(date_seas)
                dict_seasonal[season_labels[k]]["chainages"].append(np.mean(chain_seas))
                dates_seasonal.append(date_seas)
                chainage_seasonal.append(np.mean(chain_seas))
                season_ts.append(j)
    # convert chainages to np.array (easier to manipulate than a list)
    for seas in dict_seasonal.keys():
        dict_seasonal[seas]["chainages"] = np.array(dict_seasonal[seas]["chainages"])

    return (
        dict_seasonal,
        dates_seasonal,
        np.array(chainage_seasonal),
        np.array(season_ts),
    )

def calculate_trend(dates,chainage):
    "calculate long-term trend"
    dates_ord = np.array([_.toordinal() for _ in dates])
    dates_ord = (dates_ord - np.min(dates_ord))/DAYS_IN_YEAR   
    trend, intercept, rvalue, pvalue, std_err = stats.linregress(dates_ord, chainage)
    y = dates_ord*trend+intercept
    return trend, y