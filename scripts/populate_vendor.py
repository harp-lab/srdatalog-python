'''Populate `src/srdatalog/runtime/vendor/` with the C++ headers we
need from boost / highway / RMM / spdlog.

Run after a fresh `git clone` (or via the hatchling build hook,
which calls into here automatically when assembling a wheel):

    uv run python scripts/populate_vendor.py        # fetch + extract
    uv run python scripts/populate_vendor.py --check # exit 1 if stale
    uv run python scripts/populate_vendor.py --clean # wipe vendor/

Behavior:
    - Downloads source releases from each project's GitHub release
      page (no git clones — just tarballs).
    - Extracts only the include subdirs we actually need (leaves the
      rest of each release on disk during extract, then deletes the
      tarball).
    - Idempotent: a `vendor/<dep>/.version` marker stores the version
      we have; second run skips download if the version matches.
    - Caches downloads under `~/.cache/srdatalog/downloads/` so a
      repeated `--clean && populate` is fast.

Vendor layout afterwards:
    src/srdatalog/runtime/vendor/
      boost/include/boost/{hana,atomic,unordered,...}/...
      highway/include/hwy/...
      rmm/include/{rmm,spdlog,fmt}/...
      spdlog/include/spdlog/...
'''
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "src" / "srdatalog" / "runtime" / "vendor"
CACHE = Path.home() / ".cache" / "srdatalog" / "downloads"

# ---------------------------------------------------------------------------
# Dep manifest — pinned versions matching the parent SRDatalog xmake config
# ---------------------------------------------------------------------------
#
# Each entry: (download_url, sha256, [(tar_subpath, dest_under_vendor), ...])
# `tar_subpath` is the path INSIDE the extracted tarball;
# `dest_under_vendor` is where it lands relative to VENDOR/.

DEPS: dict[str, dict] = {
  "boost": {
    "version": "1.90.0",
    "url": (
      "https://archives.boost.io/release/1.90.0/source/"
      "boost_1_90_0.tar.gz"
    ),
    "sha256": "",  # set after first manual download — see _verify_sha note
    # Ship all of boost's header tree. Previously we maintained an
    # allowlist of subdirs (~30 libs) to shave wheel size, but every
    # new #include in the runtime triggers missing transitive deps.
    # The block-list below removes a handful of known-heavy libs we
    # definitely don't link against (compiled libs like log/thread/etc
    # are unused — our runtime uses only header-only boost).
    "extract": [
      ("boost_1_90_0/boost", "boost/include/boost"),
    ],
    # Subdirs to REMOVE after extract. These are either compiled libs
    # we don't use (log, thread, locale, ...) or header-only libs
    # irrelevant to datalog (python, graph, geometry, ...). Removing
    # them drops the vendored boost from ~180 MB to a manageable size
    # without fragile include-graph tracking.
    "trim_drop_subdirs": [
      "log", "thread", "locale", "asio", "beast", "json",
      "graph", "python", "geometry", "polygon", "gil",
      "wave", "mpi", "program_options", "accumulators",
      "multiprecision", "compute", "contract", "units",
      "test", "stacktrace", "coroutine", "coroutine2", "fiber",
      "context", "signals2", "interprocess", "serialization",
      "archive", "dll", "iostreams", "circular_buffer",
      "multi_index", "multi_array", "xpressive", "spirit",
      "phoenix", "proto", "logic", "icl", "flyweight",
      "heap", "bimap", "msm", "property_tree", "url",
      "convert", "leaf", "nowide", "parameter", "poly_collection",
      "uuid", "type_erasure", "date_time", "chrono", "ratio",
      "histogram", "safe_numerics", "sort", "pfr", "hof",
      "lambda", "lambda2", "local_function", "scope_exit",
      "typeof", "process", "outcome", "variant", "variant2",
      "disjoint_sets",
    ],
  },
  "highway": {
    "version": "1.3.0",
    "url": (
      "https://github.com/google/highway/archive/refs/tags/"
      "1.3.0.tar.gz"
    ),
    "sha256": "",
    "extract": [
      ("highway-1.3.0/hwy", "highway/include/hwy"),
    ],
  },
  "rmm": {
    "version": "v24.12.01",
    "url": (
      "https://github.com/rapidsai/rmm/archive/refs/tags/"
      "v24.12.01.tar.gz"
    ),
    "sha256": "",
    "extract": [
      ("rmm-24.12.01/include/rmm", "rmm/include/rmm"),
    ],
  },
  "spdlog": {
    "version": "v1.17.0",
    "url": (
      "https://github.com/gabime/spdlog/archive/refs/tags/"
      "v1.17.0.tar.gz"
    ),
    "sha256": "",
    "extract": [
      ("spdlog-1.17.0/include/spdlog", "spdlog/include/spdlog"),
    ],
  },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
  '''Cached download. Skips if `dest` already exists.'''
  if dest.exists():
    print(f"  cached: {dest.name} ({dest.stat().st_size // 1024} KB)")
    return
  dest.parent.mkdir(parents=True, exist_ok=True)
  print(f"  fetching: {url}")
  with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
    shutil.copyfileobj(resp, f)
  print(f"  → {dest} ({dest.stat().st_size // 1024} KB)")


def _verify_sha(path: Path, expected: str) -> bool:
  '''Return True iff the file's sha256 matches `expected`. Empty
  `expected` is treated as "skip check" — useful while pinning new
  deps, but should be filled in for production.'''
  if not expected:
    return True
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 16), b""):
      h.update(chunk)
  actual = h.hexdigest()
  if actual != expected:
    print(f"  [error] sha256 mismatch: expected {expected}, got {actual}")
    return False
  return True


def _extract(tar_path: Path, mappings: list[tuple[str, str]],
             stage_dir: Path) -> None:
  '''Extract only the listed subpaths from the tarball into VENDOR.
  `stage_dir` is a temp dir for the full extract before we copy out.'''
  shutil.rmtree(stage_dir, ignore_errors=True)
  stage_dir.mkdir(parents=True, exist_ok=True)
  print(f"  extracting: {tar_path.name}")
  with tarfile.open(tar_path) as tar:
    # Filter members to just what we asked for (saves disk + time).
    wanted_prefixes = tuple(src for src, _ in mappings)
    members = [m for m in tar.getmembers()
               if any(m.name.startswith(p) for p in wanted_prefixes)]
    tar.extractall(stage_dir, members=members,
                   filter="data" if sys.version_info >= (3, 12) else None)
  for src, dst in mappings:
    src_path = stage_dir / src
    dst_path = VENDOR / dst
    if dst_path.exists():
      shutil.rmtree(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_path, dst_path)
  shutil.rmtree(stage_dir, ignore_errors=True)


def _trim_boost(drop_subdirs: list[str]) -> None:
  '''Boost is huge; drop the subdirs we know we don't need.

  Deny-list approach: we keep everything by default (avoids the
  whack-a-mole of tracking transitive include graphs) and only drop
  a fixed list of boost libraries that are either compiled-lib-only
  or irrelevant to datalog (log, python, geometry, ...). Top-level
  `boost/*.hpp` umbrella headers are always kept.
  '''
  boost_root = VENDOR / "boost" / "include" / "boost"
  if not boost_root.is_dir():
    return
  drop_set = set(drop_subdirs)
  removed = 0
  for entry in list(boost_root.iterdir()):
    if entry.is_file():
      continue
    if entry.name in drop_set:
      shutil.rmtree(entry)
      removed += 1
  print(f"  dropped {removed} unused boost subdirs")


def _stamp(name: str, version: str) -> None:
  marker = VENDOR / name / ".version"
  marker.parent.mkdir(parents=True, exist_ok=True)
  marker.write_text(version + "\n")


def _stamp_matches(name: str, version: str) -> bool:
  marker = VENDOR / name / ".version"
  return marker.exists() and marker.read_text().strip() == version


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def populate(force: bool = False) -> None:
  '''Fetch + extract every dep in `DEPS`. Idempotent — if the
  `.version` marker matches, that dep is skipped unless `force=True`.
  '''
  CACHE.mkdir(parents=True, exist_ok=True)
  VENDOR.mkdir(parents=True, exist_ok=True)
  for name, info in DEPS.items():
    print(f"\n[{name}] target version {info['version']}")
    if not force and _stamp_matches(name, info["version"]):
      print(f"  already at {info['version']}, skipping")
      continue

    tar = CACHE / f"{name}-{info['version']}.tar.gz"
    _download(info["url"], tar)
    if not _verify_sha(tar, info["sha256"]):
      sys.exit(f"sha256 check failed for {tar}")
    _extract(tar, info["extract"], CACHE / f"_stage_{name}")

    if name == "boost":
      _trim_boost(info["trim_drop_subdirs"])

    _stamp(name, info["version"])
  size_mb = sum(f.stat().st_size for f in VENDOR.rglob("*") if f.is_file()) / 1024 / 1024
  fcount = sum(1 for f in VENDOR.rglob("*") if f.is_file())
  print(f"\nvendor/ ready: {fcount} files, {size_mb:.1f} MB")


def check() -> int:
  '''Return 0 if every dep matches its target version, 1 otherwise.
  Used by CI / contributor pre-commit to detect a stale vendor/.
  '''
  stale: list[str] = []
  for name, info in DEPS.items():
    if not _stamp_matches(name, info["version"]):
      stale.append(name)
  if stale:
    print(f"vendor/ stale: {', '.join(stale)} (run scripts/populate_vendor.py)")
    return 1
  print("vendor/ up-to-date")
  return 0


def clean() -> None:
  shutil.rmtree(VENDOR, ignore_errors=True)
  print(f"removed {VENDOR}")


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--force", action="store_true",
                 help="re-fetch even if version marker matches")
  p.add_argument("--check", action="store_true",
                 help="exit 1 if any dep is stale (no fetch)")
  p.add_argument("--clean", action="store_true",
                 help="wipe vendor/ and exit")
  args = p.parse_args()

  if args.clean:
    clean()
    return 0
  if args.check:
    return check()
  populate(force=args.force)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
