"""Assert the built sdist and wheel contain exactly what they should.

Run after `python -m build`:

    python ci/check_dist.py dist/

Source-tree tests never see packaging bugs, so this is the only guard between a mis-specified
glob and a broken release. What ships is decided by:

  * the layout -- [tool.setuptools.packages.find] searches only `where = ["src"]`, so the
    root-level training inputs and notebook cannot be packaged at all, which is why no
    exclude rules are needed for them;
  * MANIFEST.in, for the sdist -- `graft src` takes everything under src/, which is how
    scratch modules and the 130 MB SAR model can ride along;
  * [tool.setuptools.package-data] -- the classifiers under classification/models/.

Exit status is 0 when everything checks out, 1 with a description of what is wrong otherwise.
"""

from __future__ import annotations

import fnmatch
import glob
import os
import sys
import tarfile
import zipfile

# Every module that has to be importable from an installed wheel. `coastsat/__init__.py` is
# empty, so its presence proves nothing on its own -- the submodules are the package.
REQUIRED_MODULES = [
    "coastsat/__init__.py",
    "coastsat/SDS_classify.py",
    "coastsat/SDS_download.py",
    "coastsat/SDS_plotting.py",
    "coastsat/SDS_preprocess.py",
    "coastsat/SDS_sar_model.py",
    "coastsat/SDS_shoreline.py",
    "coastsat/SDS_tools.py",
    "coastsat/SDS_transects.py",
    "coastsat/gdal_merge.py",
]

# The classifiers SDS_shoreline.load_model() reaches for at runtime. load_model() appends
# "_new" for any scikit-learn newer than 0.20, so both eras have to ship.
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

# SDS_shoreline does `from coastsat.classification import models` and loads the classifiers
# beneath it -- the only part of classification/ the library needs.
REQUIRED_DATA = [
    "coastsat/classification/__init__.py",
    "coastsat/classification/models/__init__.py",
] + [
    f"coastsat/classification/models/NN_4classes_{stem}.pkl"
    for stem in CLASSIFIER_STEMS
]

FORBIDDEN = [
    # SAR_3_band_model.onnx is ~130 MB, over PyPI's per-file limit.
    ("*.onnx", "the SAR model is fetched at runtime, never shipped"),
    # Scratch modules that `graft src` would otherwise happily package.
    ("*_no_use.py", "scratch module"),
    ("coastsat/tests/*", "tests are not part of the distribution"),
    # Root-level and git-only. These catch a regression that moves either back under src/,
    # where package discovery would find it again. Leading `*` because fnmatch spans "/".
    (
        "*training_data/*",
        "training_data is git-only and must not ship at all",
    ),
    (
        "*training_sites/*",
        "training_sites is git-only and must not ship at all",
    ),
    (
        "*.ipynb",
        "the notebooks require a checkout and are not part of the distribution",
    ),
]

# The same rule for the sdist. "git only" means absent from both artifacts, and the sdist
# half is enforced by MANIFEST.in rather than by pyproject.toml, so it needs its own check
# -- a wheel-only assertion would pass happily while `pip install` from source pulled 32 MB.
SDIST_FORBIDDEN = [
    ("*.onnx", "the SAR model is fetched at runtime, never shipped"),
    # MANIFEST.in includes nothing outside src/, so these only fire if a broader directive
    # is added there. Match at any depth.
    ("*/training_data/*", "training_data is git-only and must not ship at all"),
    ("*/training_sites/*", "training_sites is git-only and must not ship at all"),
    # setuptools adds tests/ by default, so this one guards a MANIFEST.in `prune`
    ("*/tests/*", "tests are not part of the distribution"),
    (
        "*.ipynb",
        "the notebooks require a checkout and are not part of the distribution",
    ),
]


# Set a ceiling below 33 MB limit so that a silent size regression is turned into a failed build.
MAX_WHEEL_MB = 5.0

SDIST_REQUIRED = ["pyproject.toml", "LICENSE.md", "README.md", "PKG-INFO"]


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    sys.exit(1)


def main(dist_dir: str = "dist") -> None:
    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    sdists = glob.glob(os.path.join(dist_dir, "*.tar.gz"))
    if len(wheels) != 1:
        fail(
            f"expected exactly one wheel in {dist_dir}/, found {len(wheels)}: {wheels}"
        )
    if len(sdists) != 1:
        fail(
            f"expected exactly one sdist in {dist_dir}/, found {len(sdists)}: {sdists}"
        )
    (wheel,) = wheels
    (sdist,) = sdists

    names = set(zipfile.ZipFile(wheel).namelist())

    missing = [n for n in REQUIRED_MODULES + REQUIRED_DATA if n not in names]
    if missing:
        fail("missing from the wheel:\n  " + "\n  ".join(sorted(missing)))

    for pattern, reason in FORBIDDEN:
        hits = sorted(n for n in names if fnmatch.fnmatch(n, pattern))
        if hits:
            fail(f"{pattern} must not ship ({reason}):\n  " + "\n  ".join(hits))

    # A classifier truncated by a bad LFS or checkout step would still satisfy the presence
    # check above; the smallest real one is ~86 KB.
    for stem in CLASSIFIER_STEMS:
        entry = f"coastsat/classification/models/NN_4classes_{stem}.pkl"
        size = zipfile.ZipFile(wheel).getinfo(entry).file_size
        if size < 50_000:
            fail(f"{entry} looks truncated: {size} bytes")

    size_mb = os.path.getsize(wheel) / 1e6
    if size_mb > MAX_WHEEL_MB:
        fail(
            f"wheel is {size_mb:.1f} MB, over the {MAX_WHEEL_MB} MB ceiling -- check that "
            f"training_data/ and training_sites/ are still at the repository root rather "
            f"than under src/, where package discovery would pick them up again"
        )

    sdist_names = tarfile.open(sdist).getnames()
    for want in SDIST_REQUIRED:
        if not any(n.endswith("/" + want) for n in sdist_names):
            fail(f"{want} missing from the sdist")

    for pattern, reason in SDIST_FORBIDDEN:
        hits = sorted(n for n in sdist_names if fnmatch.fnmatch(n, pattern))
        if hits:
            fail(
                f"{pattern} must not ship in the sdist ({reason}):\n  "
                + "\n  ".join(hits)
            )

    print(f"wheel  {os.path.basename(wheel)}  {size_mb:.2f} MB  {len(names)} entries")
    print(f"sdist  {os.path.basename(sdist)}  {len(sdist_names)} entries")
    print("distribution contents OK")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dist")
