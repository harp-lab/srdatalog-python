"""Public fact acquisition must not trust a cached or unsafe archive."""

import hashlib
import stat
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from doop_suite.catalog import extract_archive, fetch_archive


def archive_spec(path):
  return {
    'name': 'sample',
    'archive_bytes': path.stat().st_size,
    'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
  }


@pytest.mark.parametrize('kind', ['traversal', 'symlink'])
def test_extraction_rejects_members_outside_fact_contract(tmp_path, kind):
  archive = tmp_path / 'sample.zip'
  with zipfile.ZipFile(archive, 'w') as stream:
    stream.writestr('sample/MainClass.facts', 'Main\n')
    if kind == 'traversal':
      stream.writestr('sample/../../escaped.facts', 'bad')
    else:
      entry = zipfile.ZipInfo('sample/Alias.facts')
      entry.create_system = 3
      entry.external_attr = (stat.S_IFLNK | 0o777) << 16
      stream.writestr(entry, '../../escaped.facts')
  with pytest.raises(ValueError):
    extract_archive(archive, archive_spec(archive), tmp_path / 'data')
  assert not (tmp_path / 'escaped.facts').exists()
  assert not (tmp_path / 'data/raw/sample').exists()


def test_existing_archive_with_wrong_bytes_is_not_reused(tmp_path):
  cached = tmp_path / 'sample.zip'
  cached.write_bytes(b'correct')
  specification = archive_spec(cached)
  cached.write_bytes(b'changed')
  with pytest.raises(ValueError, match='SHA256'):
    fetch_archive(specification, tmp_path / 'data', tmp_path)
  assert cached.read_bytes() == b'changed'
