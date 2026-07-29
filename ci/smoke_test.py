"""Import an installed coastsat and exercise the package data it needs at runtime.

    python ci/smoke_test.py                      # against whatever is importable
    python ci/smoke_test.py --require-installed  # additionally insist it came from a wheel

The release gate runs the second form from outside the checkout, so `--require-installed`
catches the case where a stray sys.path entry let src/ satisfy the imports and the wheel was
never really tested. Locally the package is installed editable, so use the plain form.

`import coastsat` on its own proves nothing -- src/coastsat/__init__.py is empty. Every real
module has to be imported by name, and the classifiers have to be located the same way
SDS_shoreline.load_model() locates them rather than by a path we guessed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

MODULES = [
    "coastsat",
    "coastsat.SDS_tools",
    "coastsat.SDS_preprocess",
    "coastsat.SDS_download",
    "coastsat.SDS_shoreline",
    "coastsat.SDS_transects",
    "coastsat.SDS_plotting",
    "coastsat.SDS_classify",
    "coastsat.SDS_sar_model",
    "coastsat.gdal_merge",
    # SDS_shoreline does `from coastsat.classification import models, training_data,
    # training_sites`, so all three have to resolve from the installed wheel. Whether they do
    # depends on package discovery agreeing with the package-data globs, which is exactly the
    # kind of thing that only breaks once it is packaged.
    "coastsat.classification",
    "coastsat.classification.models",
    "coastsat.classification.training_data",
    "coastsat.classification.training_sites",
]

CLASSIFIER_STEMS = [
    "S2",
    "S2_new",
    "Landsat",
    "Landsat_new",
    "Landsat_dark",
    "Landsat_dark_new",
    "Landsat_bright",
    "Landsat_bright_new",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="fail unless coastsat was imported from site-packages rather than a checkout",
    )
    args = parser.parse_args()

    for name in MODULES:
        module = importlib.import_module(name)
        location = getattr(module, "__file__", None) or "<namespace package>"
        print(f"ok  {name:44s} {location}")

    import coastsat

    if args.require_installed and "site-packages" not in (coastsat.__file__ or ""):
        sys.exit(f"imported the checkout, not the installed wheel: {coastsat.__file__}")

    # The exact lookup load_model() performs (SDS_shoreline.py:833).
    from coastsat import SDS_shoreline

    models_dir = SDS_shoreline.get_model_locations()
    print(f"\nclassifier directory: {models_dir}")
    for stem in CLASSIFIER_STEMS:
        path = os.path.join(models_dir, f"NN_4classes_{stem}.pkl")
        if not os.path.isfile(path):
            sys.exit(f"classifier missing from the installed package: {path}")
        if os.path.getsize(path) < 50_000:
            sys.exit(f"classifier looks truncated: {path}")
    print(f"all {len(CLASSIFIER_STEMS)} classifiers present")

    # The .kml training sites are loaded through importlib.resources, so resolve them the
    # same way rather than by joining paths.
    import importlib.resources as resources

    from coastsat.classification import training_sites

    sites_dir = os.path.abspath(resources.files(training_sites))
    for kml in ("BYRON.kml", "NEWCASTLE.kml", "SAWTELL.kml"):
        if not os.path.isfile(os.path.join(sites_dir, kml)):
            sys.exit(f"training site missing: {os.path.join(sites_dir, kml)}")
    print(f"training sites present in {sites_dir}")

    print("\nsmoke test OK")


if __name__ == "__main__":
    main()
