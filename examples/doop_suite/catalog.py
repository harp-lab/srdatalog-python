"""Pinned public DOOP facts; dataset identity is independent of workload tier."""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path, PurePosixPath

CATALOG = Path(__file__).with_name('datasets.json')


def digest(path: Path) -> str:
  result = hashlib.sha256()
  with path.open('rb') as stream:
    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
      result.update(chunk)
  return result.hexdigest()


def load_catalog() -> dict:
  return json.loads(CATALOG.read_text())


def select_datasets(names: list[str] | None = None, tier: str | None = None) -> list[dict]:
  catalog = load_catalog()
  datasets = catalog['datasets']
  known = {row['name'] for row in datasets}
  if names and (unknown := set(names) - known):
    raise ValueError(f'Unknown DOOP datasets: {sorted(unknown)}')
  if tier is not None and tier not in catalog['tiers']:
    raise ValueError(f'Unknown DOOP tier: {tier}')
  selected = [
    row
    for row in datasets
    if (not names or row['name'] in names) and (tier is None or row['tier'] == tier)
  ]
  if not selected:
    raise ValueError('Selection contains no datasets')
  return selected


def check_archive(path: Path, dataset: dict) -> None:
  if path.stat().st_size != dataset['archive_bytes']:
    raise ValueError(f'Archive size mismatch: {path}')
  if digest(path) != dataset['archive_sha256']:
    raise ValueError(f'Archive SHA256 mismatch: {path}')


def fetch_archive(dataset: dict, root: Path, archive_cache: Path | None = None) -> Path:
  """Reuse only byte-verified archives; never modify a supplied external cache."""
  root = root.resolve()
  name = dataset['name'] + '.zip'
  if archive_cache is not None and (cached := archive_cache / name).is_file():
    check_archive(cached, dataset)
    return cached.resolve()
  downloads = root / 'downloads'
  downloads.mkdir(parents=True, exist_ok=True)
  target = downloads / name
  if target.exists():
    check_archive(target, dataset)
    return target
  with tempfile.TemporaryDirectory(prefix=f'.{dataset["name"]}-', dir=downloads) as temporary:
    partial = Path(temporary) / name
    request = urllib.request.Request(dataset['url'], headers={'User-Agent': 'srdatalog-doop-suite'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('xb') as stream:
      shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
    check_archive(partial, dataset)
    partial.rename(target)
  return target


def verify_raw(archive: Path, dataset: dict, directory: Path) -> None:
  """Check a reused extraction against the pinned archive's member CRCs."""
  check_archive(archive, dataset)
  expected = set()
  with zipfile.ZipFile(archive) as source:
    for entry in source.infolist():
      if entry.is_dir():
        continue
      name = PurePosixPath(entry.filename)
      if len(name.parts) != 2 or name.parts[0] != dataset['name'] or name.suffix != '.facts':
        raise ValueError(f'Unexpected DOOP archive member: {entry.filename}')
      path = directory / name.name
      expected.add(name.name)
      if path.is_symlink() or path.stat().st_size != entry.file_size:
        raise ValueError(f'Extracted fact size/type mismatch: {path}')
      crc = 0
      with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
          crc = zlib.crc32(chunk, crc)
      if crc != entry.CRC:
        raise ValueError(f'Extracted facts changed: {path}')
  if {path.name for path in directory.iterdir()} != expected:
    raise ValueError(f'Unexpected or missing extracted facts in {directory}')


def extract_archive(archive: Path, dataset: dict, root: Path) -> Path:
  """Extract a verified archive into a new directory, rejecting links and traversal."""
  check_archive(archive, dataset)
  raw = root.resolve() / 'raw'
  raw.mkdir(parents=True, exist_ok=True)
  target = raw / dataset['name']
  if target.exists():
    raise FileExistsError(f'Refusing to overwrite extracted facts: {target}')
  with tempfile.TemporaryDirectory(prefix=f'.{dataset["name"]}-', dir=raw) as temporary:
    staging = Path(temporary)
    with zipfile.ZipFile(archive) as source:
      seen = set()
      for entry in source.infolist():
        name = PurePosixPath(entry.filename)
        mode = entry.external_attr >> 16
        if (
          name.is_absolute()
          or '..' in name.parts
          or '\\' in entry.filename
          or not name.parts
          or name.parts[0] != dataset['name']
          or stat.S_ISLNK(mode)
          or entry.filename in seen
        ):
          raise ValueError(f'Unsafe or duplicate archive member: {entry.filename}')
        seen.add(entry.filename)
        destination = staging.joinpath(*name.parts)
        if entry.is_dir():
          destination.mkdir(parents=True, exist_ok=True)
        else:
          if len(name.parts) != 2 or name.suffix != '.facts':
            raise ValueError(f'Unexpected DOOP archive member: {entry.filename}')
          destination.parent.mkdir(parents=True, exist_ok=True)
          with source.open(entry) as stream, destination.open('xb') as output:
            shutil.copyfileobj(stream, output, length=8 * 1024 * 1024)
      if not (staging / dataset['name'] / 'MainClass.facts').is_file():
        raise ValueError('Archive has no MainClass.facts')
    (staging / dataset['name']).rename(target)
  return target


def verify_prepared(directory: Path, dataset: dict) -> dict:
  """Reused prepared inputs must retain their exact provenance, metadata and tuples."""
  manifest = json.loads((directory / 'manifest.json').read_text())
  if manifest.get('provenance', {}).get('archive_sha256') != dataset['archive_sha256']:
    raise ValueError(f'Prepared dataset has different archive provenance: {directory}')
  if json.loads((directory / 'meta.json').read_text()) != manifest['meta']:
    raise ValueError(f'Prepared metadata changed: {directory}')
  for name, info in manifest['relations'].items():
    path = directory / (name + '.csv')
    if path.stat().st_size != info['bytes'] or digest(path) != info['sha256']:
      raise ValueError(f'Prepared input changed: {path}')
  return manifest
