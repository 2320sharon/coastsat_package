"""
ONNX land/water segmentation for Sentinel-1 imagery.

Runs the packaged U-Net + ResNet-50 model (``SAR_3_band_model.onnx``) on the
[VV, VH, VV-VH] dB composite that SDS_tools.build_sar_composite() produces, and returns
a boolean water mask for shoreline extraction.

The ONNX graph starts at the *normalized* input tensor: every preprocessing step lives
here, in this module. Getting one wrong does not crash, it silently degrades the
prediction, so each step below names the section of the model's inference guide it
implements. The authoritative constants are read from the .onnx file's own
metadata_props at load time (see read_model_spec) rather than hardcoded, so swapping in
a retrained model cannot silently desynchronize them.

Preprocessing, in order (guide section 2):
    1-4. read VV/VH on a common grid + combined validity mask   -> the caller does this
         via SDS_tools.read_sar_image / read_sar_valid_mask
    5.   speckle filter                                          -> none for this model
    6.   derive VV-VH before the invalid-pixel fill              -> build_sar_composite
    7.   stack as [VV, VH, VV-VH]                                -> build_sar_composite
    8-10. fill invalid with the per-channel mean, normalize,
          pad bottom/right to a multiple of the output stride    -> build_model_input

The image is never resized: the model was trained at the native ~10 m ground sample
distance and rescaling a scene measurably degrades shoreline accuracy.
"""

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from coastsat import SDS_tools

logger = logging.getLogger(__name__)

# Filename of the default model. The file is not shipped in the wheel (it is ~130 MB,
# over PyPI's file-size limit): it is downloaded on first use into pooch's OS cache dir
# by fetch_default_sar_model(), unless a copy already sits in
# coastsat/classification/models (a dev checkout) or settings['sar_model_path'] points
# elsewhere.
DEFAULT_SAR_MODEL_FILENAME = "SAR_3_band_model.onnx"

# Where the default model is downloaded from when it is not on disk. The hash pins the
# exact upload; pooch verifies it after download, so a tampered or truncated file is
# rejected rather than silently producing bad masks.
SAR_MODEL_CONFIG = {
    "model_name": DEFAULT_SAR_MODEL_FILENAME,
    "url": (
        "https://huggingface.co/2320sharon/SAR_3_band_model/resolve/main/"
        "SAR_3_band_model.onnx?download=true"
    ),
    "sha256": "sha256:8f1e5a6e2a9a50a1e1164ea0172a8545a9ca6dbf741b9133fcb9d5538508bccf",
}

# P(water) at or above which a pixel is classified as water. 0.5 reproduces argmax over
# the two logits; raise it for higher precision on the water class.
WATER_THRESHOLD = 0.5

# Class codes the model was trained with.
LAND, WATER = 0, 1

# metadata_props keys the model must declare. They say what the *model* was trained on
# rather than what this module does, and nothing at run time can catch a wrong value: the
# graph accepts any 3-channel tensor and returns a plausible-looking mask, so running a
# retrained model with another model's channel order or normalization silently degrades
# every shoreline it produces. A model that omits them is rejected instead (the run then
# falls back to Otsu, see SDS_shoreline.load_sar_segmenter).
REQUIRED_METADATA_KEYS = ("channel_order", "normalization_mean", "normalization_std")

# Defaults for the keys that describe how to *run* the graph rather than what it was
# trained on. Guessing these is safe because a wrong guess fails loudly: _validate_spec
# rejects a preprocessing contract this module does not implement, and onnxruntime
# rejects an input padded to the wrong multiple. They match SAR_3_band_model.onnx
# (run s1_water_3band_7_8_26_50_epochs).
DEFAULT_SPEC: Dict[str, Any] = {
    "output_stride": 32,
    "nodata": 255,
    "input_scale": "dB",
    "speckle_filter": {"type": "none"},
    "classes": {"0": "land", "1": "water"},
}

# metadata_props key -> spec key. The .onnx spells some of them differently.
_METADATA_KEYS = {
    "channel_order": "channel_order",
    "normalization_mean": "mean",
    "normalization_std": "std",
    "speckle_filter": "speckle_filter",
    "output_stride": "output_stride",
    "nodata": "nodata",
    "input_scale": "input_scale",
    "classes": "classes",
}

# Name of the graph output holding P(water). Resolved by name rather than by position so
# a re-export that reorders the outputs cannot silently return the logits instead.
_WATER_PROB_OUTPUT = "water_prob"
_LOGITS_OUTPUT = "logits"

# One session per model path: the file is ~130 MB and extract_shorelines loops over scenes.
_SESSION_CACHE: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


class SarModelUnavailable(RuntimeError):
    """
    The model cannot be used, so the caller should fall back to Otsu thresholding.

    Raised for every "this scene or this environment is not usable" condition -
    onnxruntime not installed, the .onnx missing, a single-polarization scene, a model
    whose preprocessing contract does not match this code - so callers have a single
    exception type to catch.
    """


def _model_cache_dir() -> str:
    """Directory the default model is downloaded into. Never raises: this runs during
    plain path resolution (including test collection), where pooch may be absent."""
    try:
        import pooch  # lazy: only the download path needs pooch

        return str(pooch.os_cache("coastsat"))
    except ImportError:
        return os.path.join(os.path.expanduser("~"), ".cache", "coastsat")


def get_sar_model_path(settings: Optional[Dict[str, Any]] = None) -> str:
    """
    Returns the path of the SAR segmentation model to run. Never touches the network.

    Resolution order: settings['sar_model_path'] override, then a copy sitting in
    coastsat/classification/models (a dev checkout), then the pooch cache path where
    fetch_default_sar_model() downloads to.

    Arguments:
    -----------
    settings: dict, optional
        may contain 'sar_model_path' to override the default model

    Returns:
    -----------
    str: path to the .onnx file (not checked for existence here)

    """
    if settings:
        override = settings.get("sar_model_path")
        if override:
            return os.path.abspath(str(override))

    # imported here to avoid a circular import: SDS_shoreline imports this module
    from coastsat.SDS_shoreline import get_model_locations

    bundled = os.path.join(get_model_locations(), DEFAULT_SAR_MODEL_FILENAME)
    if os.path.isfile(bundled):
        return bundled

    return os.path.abspath(
        os.path.join(_model_cache_dir(), DEFAULT_SAR_MODEL_FILENAME)
    )


def fetch_default_sar_model() -> str:
    """
    Downloads the default model into the pooch cache and returns its path.

    Skips the download when the cached copy is already present (pooch checks the
    hash). Every failure - pooch not installed, network down, HTTP error, hash
    mismatch, disk full - raises SarModelUnavailable so callers fall back to Otsu.

    Returns:
    -----------
    str: absolute path of the verified .onnx file

    """
    try:
        import pooch
    except ImportError as exc:
        raise SarModelUnavailable(
            "pooch is not installed, so the SAR segmentation model cannot be downloaded"
        ) from exc

    try:
        downloaded = pooch.retrieve(
            url=SAR_MODEL_CONFIG["url"],
            known_hash=SAR_MODEL_CONFIG["sha256"],
            fname=SAR_MODEL_CONFIG["model_name"],
            path=pooch.os_cache("coastsat"),
            progressbar=True,  # renders with tqdm, already a dependency
        )
    except Exception as exc:
        raise SarModelUnavailable(
            f"could not download the SAR segmentation model from "
            f"{SAR_MODEL_CONFIG['url']}: {exc}"
        ) from exc

    return os.path.abspath(str(downloaded))


def read_model_spec(session) -> Dict[str, Any]:
    """
    Reads the preprocessing contract from the model's embedded metadata.

    The .onnx is self-describing: its metadata_props carry the channel order,
    normalization constants, output stride and speckle filter the model was trained
    with. Reading them here means a retrained model brings its own constants instead of
    silently disagreeing with hardcoded ones, so the constants it was trained on
    (REQUIRED_METADATA_KEYS) must be present. The keys that only describe how to run the
    graph fall back to DEFAULT_SPEC.

    Arguments:
    -----------
    session: onnxruntime.InferenceSession
        an open session on the model

    Returns:
    -----------
    dict: 'channel_order', 'mean', 'std', 'output_stride', 'nodata', 'input_scale',
        'speckle_filter' and 'classes'

    Raises:
    -----------
    SarModelUnavailable
        if the model does not declare the constants it was trained with, or if its
        preprocessing contract is not the one this module implements

    """
    spec = dict(DEFAULT_SPEC)

    try:
        metadata = session.get_modelmeta().custom_metadata_map or {}
    except Exception:  # a stub session or a model exported without metadata
        metadata = {}

    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    if missing:
        raise SarModelUnavailable(
            f"the model does not declare {missing} in its metadata_props, so the "
            "constants it was trained with are unknown; re-export it with those keys "
            "rather than running it on another model's channel order and normalization"
        )

    for meta_key, spec_key in _METADATA_KEYS.items():
        if meta_key not in metadata:
            continue
        raw = metadata[meta_key]
        try:
            spec[spec_key] = json.loads(raw)
        except (TypeError, ValueError):
            # plain strings such as input_scale are not JSON
            spec[spec_key] = raw

    try:
        spec["output_stride"] = int(spec["output_stride"])
        spec["mean"] = np.asarray(spec["mean"], dtype=np.float32)
        spec["std"] = np.asarray(spec["std"], dtype=np.float32)
    except (TypeError, ValueError) as exc:
        # a malformed export, e.g. normalization_mean written as a bare string
        raise SarModelUnavailable(
            f"the model's metadata_props cannot be read as numbers: {exc}"
        ) from exc

    _validate_spec(spec)

    return spec


def _validate_spec(spec: Dict[str, Any]) -> None:
    """Rejects a model whose preprocessing contract differs from what this module does."""
    channel_order = [str(band) for band in spec["channel_order"]]
    expected = list(SDS_tools.SAR_COMPOSITE_BANDS)
    if channel_order != expected:
        raise SarModelUnavailable(
            f"model expects channels {channel_order} but this code builds {expected}; "
            "feeding it the composite would silently transpose the bands"
        )

    if len(spec["mean"]) != len(expected) or len(spec["std"]) != len(expected):
        raise SarModelUnavailable(
            f"model has {len(spec['mean'])} means and {len(spec['std'])} stds "
            f"for {len(expected)} channels"
        )

    if np.any(spec["std"] == 0):
        raise SarModelUnavailable(f"model has a zero normalization std: {spec['std']}")

    speckle_filter = spec.get("speckle_filter") or {}
    filter_type = (
        speckle_filter.get("type") if isinstance(speckle_filter, dict) else speckle_filter
    )
    if str(filter_type).lower() != "none":
        # the Lee-family filters do not export into the graph, so they would have to be
        # implemented here; running an unfiltered scene through a filtered model is wrong
        raise SarModelUnavailable(
            f"model was trained with the '{filter_type}' speckle filter, which this "
            "code does not apply"
        )

    if str(spec.get("input_scale", "dB")).lower() != "db":
        raise SarModelUnavailable(
            f"model expects '{spec['input_scale']}' input, but the downloaded "
            "Sentinel-1 rasters are in dB"
        )

    if int(spec["output_stride"]) < 1:
        raise SarModelUnavailable(f"invalid output stride: {spec['output_stride']}")


def load_sar_model(model_path: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """
    Opens an onnxruntime session on the SAR model and reads its preprocessing spec.

    Sessions are cached per path, so calling this once per satellite in a loop is cheap.

    Arguments:
    -----------
    model_path: str, optional
        path to the .onnx file. None means the default model, which is downloaded on
        first use if it is not on disk; an explicit path is never downloaded.

    Returns:
    -----------
    tuple: (onnxruntime.InferenceSession, spec dict as returned by read_model_spec)

    Raises:
    -----------
    SarModelUnavailable
        if onnxruntime is not installed, the file is missing (or its download fails),
        or the session or its metadata cannot be used

    """
    is_default = model_path is None
    if is_default:
        model_path = get_sar_model_path()
    model_path = os.path.abspath(str(model_path))

    if model_path in _SESSION_CACHE:
        return _SESSION_CACHE[model_path]

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SarModelUnavailable(
            "onnxruntime is not installed, so the SAR segmentation model cannot be run"
        ) from exc

    if not os.path.isfile(model_path):
        if is_default:
            # onnxruntime was checked first: don't pull 130 MB into an
            # environment that cannot run the model anyway
            model_path = fetch_default_sar_model()
        else:
            # explicit sar_model_path overrides are never downloaded
            raise SarModelUnavailable(
                f"SAR segmentation model not found: {model_path}"
            )

    try:
        session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
    except Exception as exc:
        raise SarModelUnavailable(
            f"could not open the SAR segmentation model '{model_path}': {exc}"
        ) from exc

    spec = read_model_spec(session)

    _SESSION_CACHE[model_path] = (session, spec)
    return session, spec


def clear_model_cache() -> None:
    """Drops every cached session. Only needed by the tests."""
    _SESSION_CACHE.clear()


def pad_to_multiple(image: np.ndarray, stride: int) -> np.ndarray:
    """
    Pads the bottom and right edges of a [C, H, W] array with zeros (guide section 2.10).

    The encoder halves the resolution five times, so H and W must each be a multiple of
    the output stride at run time. Padding with 0.0 is neutral because the image has
    already been normalized; the caller crops the prediction back to the original size.

    Arguments:
    -----------
    image: np.ndarray
        (C, H, W) normalized image
    stride: int
        required multiple, 32 for this model

    Returns:
    -----------
    np.ndarray: (C, H + pad_h, W + pad_w) padded image

    """
    _, height, width = image.shape
    pad_h = (stride - height % stride) % stride
    pad_w = (stride - width % stride) % stride
    if not pad_h and not pad_w:
        return image
    return np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)))


def build_model_input(
    im_composite: np.ndarray, valid: np.ndarray, spec: Dict[str, Any]
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Turns the [VV, VH, VV-VH] dB composite into the model's input tensor.

    Implements steps 7 to 10 of the guide: stack (already done by build_sar_composite,
    which derives VV-VH from the raw dB bands *before* any fill), fill invalid pixels
    with the per-channel mean so they normalize to exactly 0, normalize, then pad.

    Arguments:
    -----------
    im_composite: np.ndarray
        (H, W, 3) composite in dB, channels [VV, VH, VV-VH]
    valid: np.ndarray
        (H, W) boolean, True where both polarizations hold usable data
    spec: dict
        as returned by read_model_spec

    Returns:
    -----------
    tuple:
        np.ndarray: (1, 3, Hp, Wp) float32 NCHW input tensor
        tuple: the original (H, W), for cropping the prediction back

    Raises:
    -----------
    SarModelUnavailable
        if the scene is not a 3-channel composite, i.e. it lacks VV or VH

    """
    n_channels = len(spec["channel_order"])

    if im_composite.ndim != 3 or im_composite.shape[2] != n_channels:
        raise SarModelUnavailable(
            f"the model needs a {n_channels}-channel "
            f"{list(spec['channel_order'])} composite but this scene is "
            f"{im_composite.shape}, so it is missing VV or VH"
        )
    if valid.shape != im_composite.shape[:2]:
        raise SarModelUnavailable(
            f"validity mask {valid.shape} does not match the image "
            f"{im_composite.shape[:2]}"
        )

    mean = np.asarray(spec["mean"], dtype=np.float32)
    std = np.asarray(spec["std"], dtype=np.float32)

    # (H, W, C) -> (C, H, W); copy because the fill below writes into it
    image = np.ascontiguousarray(
        np.transpose(im_composite, (2, 0, 1)).astype(np.float32)
    )

    # step 8: invalid pixels become the channel mean, i.e. exactly 0 after normalizing,
    # which is a neutral input instead of an extreme outlier
    image[:, ~valid] = mean[:, None]

    # step 9
    image = (image - mean[:, None, None]) / std[:, None, None]

    height, width = im_composite.shape[:2]
    # step 10
    image = pad_to_multiple(image, int(spec["output_stride"]))

    return image[None].astype(np.float32), (height, width)


def _water_prob_from_outputs(session, outputs) -> np.ndarray:
    """Picks P(water) out of the session outputs by name, softmaxing logits if needed."""
    names = [output.name for output in session.get_outputs()]

    if _WATER_PROB_OUTPUT in names:
        return np.asarray(outputs[names.index(_WATER_PROB_OUTPUT)])

    if _LOGITS_OUTPUT in names:
        logits = np.asarray(outputs[names.index(_LOGITS_OUTPUT)], dtype=np.float32)
        # softmax over the class axis, then keep the water channel
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return (exponentials / exponentials.sum(axis=1, keepdims=True))[:, WATER : WATER + 1]

    raise SarModelUnavailable(
        f"model has outputs {names}, none of which is "
        f"'{_WATER_PROB_OUTPUT}' or '{_LOGITS_OUTPUT}'"
    )


def segment_water(
    im_composite: np.ndarray,
    valid: np.ndarray,
    session,
    spec: Dict[str, Any],
    threshold: float = WATER_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segments water in one Sentinel-1 scene with the ONNX model.

    Arguments:
    -----------
    im_composite: np.ndarray
        (H, W, 3) [VV, VH, VV-VH] composite in dB, as returned by preprocess_image
    valid: np.ndarray
        (H, W) boolean validity mask, from SDS_tools.read_sar_valid_mask
    session: onnxruntime.InferenceSession
        an open session, from load_sar_model
    spec: dict
        the model's preprocessing spec, from load_sar_model
    threshold: float
        P(water) at or above which a pixel is water

    Returns:
    -----------
    tuple:
        np.ndarray: (H, W) boolean water mask, False at invalid pixels
        np.ndarray: (H, W) float32 P(water), NaN at invalid pixels

    Raises:
    -----------
    SarModelUnavailable
        if the scene is not a 3-channel composite or inference fails

    """
    image, (height, width) = build_model_input(im_composite, valid, spec)

    input_name = session.get_inputs()[0].name
    try:
        outputs = session.run(None, {input_name: image})
    except Exception as exc:
        raise SarModelUnavailable(f"SAR model inference failed: {exc}") from exc

    water_prob = _water_prob_from_outputs(session, outputs)

    # crop the padding back off before anything is written or thresholded
    prob = np.asarray(water_prob[0, 0, :height, :width], dtype=np.float32).copy()
    im_water = prob >= threshold

    # blank input areas must never be mistaken for a land or water call
    im_water[~valid] = False
    prob[~valid] = np.nan

    return im_water, prob
