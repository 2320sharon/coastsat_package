import os
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import gridspec, lines as mlines
from matplotlib.colors import ListedColormap


from coastsat import SDS_preprocess, SDS_tools

# Colour the reference shoreline buffer actually renders as. The buffer is drawn as an
# all-True masked array under cmap="PiYG", which normalises to 0.0 and so takes the
# bottom of the colormap - the pink end, not the green one. The legend swatch has to
# match what is on the image or it points the reader at the wrong thing.
REF_BUFFER_COLOR = "#8e0152"
REF_BUFFER_ALPHA = 0.6


def transform_shoreline_area_to_pixel_coords(
    shoreline_extraction_area: Optional[List[np.ndarray]],
    output_epsg: int,
    georef: np.ndarray,
    image_epsg: int,
    transform_func,
) -> List[np.ndarray]:
    """
    Transforms a list of world coordinates into pixel coordinates using a transformation function.

    Args:
        shoreline_extraction_area (Optional[List[np.ndarray]]):
            The input list of world coordinates (or None).
            Example:
            [array([[ 606270.9268185 , 4079016.66886246],
                    [ 606967.96694302, 4077461.14324242],
                    [ 607305.27201052, 4076390.54086919],
                    [ 607289.08625894, 4076373.12052936],
                    [ 606690.57810944, 4076373.12052936],
                    [ 606061.90757097, 4077987.18491909],
                    [ 605284.49195324, 4079756.58371796],
                    [ 604687.47197666, 4080853.12052936],
                    [ 605266.43466503, 4080853.12052936],
                    [ 606270.9268185 , 4079016.66886246]])]
        output_epsg (int): EPSG code for the world coordinate system.
        georef (np.ndarray): Georeferencing metadata used for transformation.
            Example:  [ 6.04075722e+05  1.00000000e+01  0.00000000e+00  4.08085444e+06, 0.00000000e+00 -1.00000000e+01]
        image_epsg (int): EPSG code of the image coordinate system.
            Example : 32610 for UTM zone 10N.
        transform_func (callable): Function to perform the world-to-pixel transformation.
            Example: SDS_preprocess.transform_world_coords_to_pixel_coords - default transformation function.

    Returns:
        the transformed pixel coordinates as a list of tuples (x, y) or an empty list if the input is None or empty.
        Example:
        Shoreline extraction area in pixel coordinates:
            [array([[ 2.20236444e+02,  1.83645167e+02],
                [ 2.89940456e+02,  3.39197729e+02],
                [ 3.23670963e+02,  4.46257966e+02],
                [ 3.22052388e+02,  4.48000000e+02],
                [ 2.62201573e+02,  4.48000000e+02],
                [ 1.99334519e+02,  2.86593561e+02],
                [ 1.21592957e+02,  1.09653681e+02],
                [ 6.18909597e+01, -5.82076609e-11],
                [ 1.19787229e+02, -5.82076609e-11],
                [ 2.20236444e+02,  1.83645167e+02]])]
    """
    if not shoreline_extraction_area:
        return []

    if shoreline_extraction_area is not None:
        if len(shoreline_extraction_area) == 0:
            return []

    pixel_coords = [
        transform_func(coord, output_epsg, georef, image_epsg)
        for coord in shoreline_extraction_area
    ]

    return pixel_coords


def plot_detection(
    im_ms: np.ndarray,
    im_labels: np.ndarray,
    shoreline: np.ndarray,
    image_epsg: int,
    georef: np.ndarray,
    settings: Dict[str, Any],
    date: str,
    satname: str,
    cloud_mask: Optional[np.ndarray] = None,
    im_ref_buffer: Optional[np.ndarray] = None,
    output_directory: Optional[str] = None,
    shoreline_extraction_area: Optional[np.ndarray] = None,
    is_sar: bool = False,
    band_index: int = 0,
    class_labels: Tuple[str, str] = ("Otsu class 1", "Otsu class 2"),
):
    """
    Unified function for plotting shoreline detection on SAR and optical images.
    Saves the plot and optionally prompts user input to skip/accept detections.

    Parameters:
    - im_ms: np.array - image array (SAR or multispectral)
    - im_labels: np.array - binary mask or multi-channel classification
    - shoreline: np.array - coordinates of shoreline (X, Y)
    - image_epsg: int - EPSG code of image CRS
    - georef: np.array - georeferencing array
    - settings: dict - config dictionary
    - date: str - date of image
    - satname: str - satellite name
    - cloud_mask: np.array - mask of cloud pixels (optical only)
    - im_ref_buffer: np.array - reference shoreline buffer
    - output_directory: str - path to store outputs
    - shoreline_extraction_area: list[np.array] - area outlines
    - is_sar: bool - whether the input is SAR imagery
    - band_index: int - SAR only, which channel of im_ms to display
    - class_labels: tuple[str, str] - SAR only, what im_labels == 0 and == 1 mean

    Returns
        bool: True if the user accepted the detections, False otherwise.

    """
    sitename = settings["inputs"]["sitename"]
    filepath_data = settings["inputs"]["filepath"]
    output_path = _prepare_output_dir(output_directory, filepath_data, sitename)

    sl_pix = _transform_shoreline(
        shoreline, settings["output_epsg"], image_epsg, georef
    )

    shoreline_extraction_area_pix = transform_shoreline_area_to_pixel_coords(
        shoreline_extraction_area,
        output_epsg=settings["output_epsg"],
        georef=georef,
        image_epsg=image_epsg,
        transform_func=SDS_preprocess.transform_world_coords_to_pixel_coords,
    )

    if is_sar:
        return plot_sar_detection(
            im_ms,
            im_labels,
            sl_pix,
            date,
            satname,
            im_ref_buffer,
            output_path,
            settings,
            shoreline_extraction_area=shoreline_extraction_area_pix,
            band_index=band_index,
            class_labels=class_labels,
        )
    else:
        return plot_optical_detection(
            im_ms,
            im_labels,
            cloud_mask,
            sl_pix,
            date,
            satname,
            im_ref_buffer,
            output_path,
            settings,
            sitename,
            shoreline_extraction_area=shoreline_extraction_area_pix,
        )


def _transform_shoreline(shoreline, output_epsg, image_epsg, georef):
    try:
        shoreline_proj = SDS_tools.convert_epsg(shoreline, output_epsg, image_epsg)
        return SDS_tools.convert_world2pix(shoreline_proj[:, :2], georef)
    except Exception:
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])


def _prepare_output_dir(output_dir, base_path, sitename):
    path = os.path.join(
        output_dir or os.path.join(base_path, sitename), "jpg_files", "detection"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_grayscale(im):
    im = np.nan_to_num(im, nan=0.0)
    im_min, im_max = np.percentile(im, [2, 98])
    return np.clip((im - im_min) / (im_max - im_min), 0, 1)


def determine_layout(im_shape):
    """
    Determines the appropriate subplot layout based on image aspect ratio. By default, it uses a horizontal layout.

    Args:
        im_shape (tuple): Shape of the image (height, width[, channels])

    Returns:
        gs (GridSpec): GridSpec layout
        ax_indices (tuple): ((ax1_idx), (ax2_idx), (ax3_idx)) as grid positions
        legend_settings (dict): Dict with bbox_to_anchor and loc for legends
    """
    height, width = im_shape[:2]
    if height > 2.5 * width:  # This means the image is vertical
        gs = gridspec.GridSpec(3, 1)
        ax_indices = ((0, 0), (1, 0), (2, 0))
        legend_settings = {"bbox_to_anchor": (1.05, 0.5), "loc": "center left"}
    else:  # This means the image is horizontal (default)
        gs = gridspec.GridSpec(1, 3)
        ax_indices = ((0, 0), (0, 1), (0, 2))
        legend_settings = {"bbox_to_anchor": (0.5, -0.23), "loc": "lower center"}

    return gs, ax_indices, legend_settings


def plot_sar_detection(
    im_ms: np.ndarray,
    im_labels: np.ndarray,
    sl_pix: np.ndarray,
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    settings: Dict[str, Any],
    shoreline_extraction_area: List[np.ndarray],
    band_index: int = 0,
    class_labels: Tuple[str, str] = ("Otsu class 1", "Otsu class 2"),
) -> bool:
    """
    Plots SAR detection results as three panels, left to right:

        1. the SAR imagery with the detected shoreline
        2. the same, plus the land/water segmentation the shoreline was traced from
        3. the same as 2, plus the reference shoreline buffer that restricted the
           search, overlaid as plot_optical_detection draws it

    Panel 3 repeats panel 2 on purpose: it is the only way to see whether the buffer
    cut through the segmentation, without the buffer obscuring the segmentation in the
    panel used to judge it.

    Args:
        im_ms (np.ndarray): SAR image to display, one channel per polarization.
        im_labels (np.ndarray): Labeled image (typically from Otsu thresholding).
        sl_pix (np.ndarray): Array of shoreline pixel coordinates (Nx2).
        date (str): Date string for display and file naming.
        satname (str): Satellite name for display and file naming.
        im_ref_buffer (Optional[np.ndarray]): Boolean mask of the reference shoreline buffer.
        output_path (str): Directory path to save the figure if enabled.
        settings (Dict[str, Any]): Dictionary of settings. Must contain key 'save_figure'.
        shoreline_extraction_area (list[np.ndarray]): The shoreline extraction area in pixel coordinates.
        band_index (int): Which channel (polarization) of im_ms to display.
        class_labels (tuple[str, str]): What im_labels == 0 and im_labels == 1 mean, for
            the legend. The two segmentation methods label opposite classes: Otsu marks
            the high-backscatter class (land) as 1, the model marks water as 1.

    Returns:
        bool: Always returns False. Used as a placeholder return value.
    """
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 3)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    # Reduce to a single polarization before normalising: imshow only accepts 1, 3 or 4
    # channels, and the percentiles would otherwise be taken over a mix of polarizations
    # whose dB distributions differ.
    im_band = im_ms[:, :, band_index] if im_ms.ndim == 3 else im_ms
    im_display = _normalize_grayscale(im_band)

    # A buffer covering the whole image means no reference shoreline was given, so it
    # constrains nothing. Drawing it would just tint the entire panel.
    show_ref_buffer = im_ref_buffer is not None and not np.all(im_ref_buffer)

    # Left: the imagery on its own, so the detected shoreline can be judged against it
    ax1.imshow(im_display, cmap="gray")
    _draw_sar_shoreline(ax1, sl_pix, shoreline_extraction_area)
    ax1.set_title(date)
    ax1.axis("off")

    # Middle: the segmentation the shoreline was traced from
    ax2.imshow(im_display, cmap="gray")
    _draw_sar_classes(ax2, im_labels)
    _draw_sar_shoreline(ax2, sl_pix, shoreline_extraction_area)
    ax2.set_title(satname)
    ax2.axis("off")

    # Right: the same, plus the reference shoreline buffer that restricted the search,
    # drawn the way plot_optical_detection draws it
    ax3.imshow(im_display, cmap="gray")
    _draw_sar_classes(ax3, im_labels)
    if show_ref_buffer:
        ax3.imshow(
            np.ma.masked_where(im_ref_buffer == False, im_ref_buffer),
            cmap="PiYG",
            alpha=REF_BUFFER_ALPHA,
        )
        ax3_title = "reference shoreline buffer"
    else:
        ax3_title = "no reference shoreline (whole image searched)"
    _draw_sar_shoreline(ax3, sl_pix, shoreline_extraction_area)
    ax3.set_title(ax3_title)
    ax3.axis("off")

    handles = [mlines.Line2D([], [], color="red", label="Shoreline")]
    # the colormap in _draw_sar_classes maps im_labels 0 -> yellow and 1 -> blue
    handles += [
        mpatches.Patch(color="yellow", label=class_labels[0]),
        mpatches.Patch(color="blue", label=class_labels[1]),
    ]
    if show_ref_buffer:
        handles.append(
            mpatches.Patch(
                color=REF_BUFFER_COLOR,
                alpha=REF_BUFFER_ALPHA,
                label="reference shoreline buffer",
            )
        )
    if shoreline_extraction_area:
        handles.append(
            mlines.Line2D([], [], color="#cb42f5", label="shoreline extraction area")
        )

    # loc must be given with bbox_to_anchor, otherwise it defaults to "best" and the
    # legend is placed *inside* the panel, on top of the imagery
    ax3.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10
    )

    if settings.get("save_figure", False):
        fig.savefig(
            os.path.join(output_path, f"{date}_{satname}.jpg"),
            dpi=150,
            bbox_inches="tight",
        )

    plt.close(fig)
    return False


def _draw_sar_classes(ax: matplotlib.axes.Axes, im_labels: np.ndarray) -> None:
    """Overlays the two-class segmentation; 0 -> yellow, 1 -> blue."""
    ax.imshow(im_labels.astype(int), cmap=ListedColormap(["yellow", "blue"]), alpha=0.3)


def _draw_sar_shoreline(
    ax: matplotlib.axes.Axes,
    sl_pix: np.ndarray,
    shoreline_extraction_area: List[np.ndarray],
) -> None:
    """Draws the detected shoreline and any extraction-area outlines onto an axis."""
    ax.plot(sl_pix[:, 0], sl_pix[:, 1], "r.", markersize=1)
    for area in shoreline_extraction_area:
        ax.plot(area[:, 0], area[:, 1], color="#cb42f5", markersize=1)


def plot_optical_detection(
    im_ms: np.ndarray,
    im_labels: np.ndarray,
    cloud_mask: np.ndarray,
    sl_pix: np.ndarray,
    date: str,
    satname: str,
    im_ref_buffer: Optional[np.ndarray],
    output_path: str,
    settings: Dict[str, Any],
    sitename: str,
    shoreline_extraction_area: Optional[np.ndarray],
) -> bool:
    """
    Plots optical shoreline detection including RGB composite, classified output,
    NDWI index, cloud mask, and reference buffer.

    Args:
        im_ms (np.ndarray): Multispectral image array (HxWxBands).
        im_labels (np.ndarray): Labeled output image (e.g., water/land mask).
        cloud_mask (np.ndarray): Boolean array indicating cloud-covered pixels.
        sl_pix (np.ndarray): Nx2 array of detected shoreline pixel coordinates.
        date (str): Acquisition date string.
        satname (str): Name of the satellite (for display and filename).
        im_ref_buffer (Optional[np.ndarray]): Boolean mask for reference shoreline buffer.
        output_path (str): Directory to save the output image.
        settings (Dict[str, Any]): Settings dictionary, should include 'save_figure' flag.
        sitename (str): Name of the site (for display in title).
        shoreline_extraction_area (list[np.ndarry]): The shoreline extraction area in pixel coordinates.
    Returns:
        bool: `True` if the image was skipped by user input; otherwise `False`.
    """
    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(1, 3)
    ax1, ax2, ax3 = [fig.add_subplot(gs[0, i]) for i in range(3)]

    im_rgb = SDS_preprocess.rescale_image_intensity(
        im_ms[:, :, [2, 1, 0]], cloud_mask, 99.9
    )
    im_class = _get_im_class(im_labels, im_rgb)
    ax1.imshow(np.nan_to_num(im_rgb))
    ax1.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for area in shoreline_extraction_area:
        ax1.plot(
            area[:, 0],
            area[:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax1.set_title(sitename, fontweight="bold", fontsize=16)
    ax1.axis("off")

    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax2.imshow(mask, cmap="PiYG", alpha=0.6)

    ax2.imshow(im_class)
    # plot the reference shoreline buffer
    if im_ref_buffer is not None:
        mask = np.ma.masked_where(im_ref_buffer == False, im_ref_buffer)
        ax2.imshow(mask, cmap="PiYG", alpha=0.6)
    ax2.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for area in shoreline_extraction_area:
        ax2.plot(
            area[:, 0],
            area[:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax2.set_title(date, fontweight="bold", fontsize=16)
    ax2.axis("off")

    _add_optical_legend(ax2, bool(shoreline_extraction_area))

    im_mwi = SDS_tools.nd_index(im_ms[:, :, 4], im_ms[:, :, 1], cloud_mask)
    ax3.imshow(im_mwi, cmap="bwr")
    if im_ref_buffer is not None:
        ax3.imshow(
            np.ma.masked_where(~im_ref_buffer, im_ref_buffer), cmap="PiYG", alpha=0.6
        )
    ax3.plot(sl_pix[:, 0], sl_pix[:, 1], "k.", markersize=1)
    for area in shoreline_extraction_area:
        ax3.plot(
            area[:, 0],
            area[:, 1],
            color="#cb42f5",
            markersize=1,
        )
    ax3.set_title(satname, fontweight="bold", fontsize=16)
    ax3.axis("off")

    skip_image = _handle_user_input(fig, ax1, settings)

    if settings.get("save_figure", False) and not skip_image:
        fig.savefig(
            os.path.join(output_path, f"{date}_{satname}.jpg"),
            dpi=150,
            bbox_inches="tight",
        )

    plt.close(fig)
    return skip_image


def _get_im_class(im_labels: np.ndarray, im_rgb: np.ndarray) -> np.ndarray:
    """
    Applies color overlays to an RGB image based on label masks.

    Args:
        im_labels (np.ndarray): Boolean label masks of shape (H, W, C), where each channel corresponds to a class.
        im_rgb (np.ndarray): RGB image of shape (H, W, 3) used as the base image.

    Returns:
        np.ndarray: RGB image with overlaid class colors.
    """
    cmap = plt.get_cmap("tab20c")
    colors = np.array(
        [
            cmap(5),  # sand
            [204 / 255, 1, 1, 1],  # whitewater
            [0, 91 / 255, 1, 1],  # water
        ]
    )
    im_class = np.copy(im_rgb)
    for i in range(im_labels.shape[2]):
        im_class[im_labels[:, :, i]] = colors[i][:3]
    return im_class


def _add_optical_legend(
    ax: matplotlib.axes.Axes, include_extraction_area: bool
) -> None:
    """
    Adds a legend to the provided Axes object for optical shoreline classification.

    Args:
        ax (matplotlib.axes.Axes): The axis to which the legend will be added.
        include_extraction_area (bool): Whether to include the shoreline extraction area in the legend.
    """
    handles = [
        mpatches.Patch(color=(0.894, 0.329, 0.188, 1), label="sand"),
        mpatches.Patch(color=(0.8, 1.0, 1.0, 1.0), label="whitewater"),
        mpatches.Patch(color=(0, 0.357, 1.0, 1.0), label="water"),
        mlines.Line2D([], [], color="k", label="shoreline"),
        mpatches.Patch(color="#800000", alpha=0.8, label="reference shoreline buffer"),
    ]
    if include_extraction_area:
        handles.append(
            mlines.Line2D([], [], color="#cb42f5", label="shoreline extraction area")
        )
    ax.legend(handles=handles, bbox_to_anchor=(1, 0.5), fontsize=10)


def _handle_user_input(fig, ax, settings):
    if not settings.get("check_detection", False):
        return False

    key_event = {}

    def press(event):
        key_event["pressed"] = event.key

    fig.canvas.mpl_connect("key_press_event", press)

    while True:
        ax.text(1.1, 0.9, "keep ⇨", size=12, transform=ax.transAxes)
        ax.text(-0.1, 0.9, "⇦ skip", size=12, transform=ax.transAxes)
        ax.text(0.5, 0, "<esc> to quit", size=12, transform=ax.transAxes)
        plt.draw()
        plt.waitforbuttonpress()

        if key_event.get("pressed") == "right":
            return False
        elif key_event.get("pressed") == "left":
            return True
        elif key_event.get("pressed") == "escape":
            plt.close()
            raise StopIteration("User cancelled checking shoreline detection")
