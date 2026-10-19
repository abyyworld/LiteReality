"""A cached asset must not hide failure of its edited Blender recipe."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from litereality_agent.room_ops.compile import build_from_room as build


@pytest.mark.parametrize('returncode, output', [(0, None), (1, b'glTF' + b'\0' * 32), (0, b'bad')])
def test_failed_regeneration_preserves_old_cache_but_returns_failure(tmp_path, monkeypatch, returncode, output):
    (tmp_path / 'object.py').write_text('raise RuntimeError("broken recipe")')
    cache = tmp_path / 'Object.glb'
    cache.write_bytes(b'previous asset')
    def run(cmd, **kwargs):
        assert cmd[cmd.index('--python-exit-code') + 1] == '1'
        assert Path(cmd[-1]) != cache
        if output is not None:
            Path(cmd[-1]).write_bytes(output)
        return SimpleNamespace(returncode=returncode, stderr=b'failed')
    monkeypatch.setattr(build.subprocess, 'run', run)
    assert not build.run_object_py(tmp_path, 'Object', True, cache, tmp_path, 'blender', True)
    assert cache.read_bytes() == b'previous asset'


def test_successful_regeneration_replaces_cached_geometry(tmp_path, monkeypatch):
    (tmp_path / 'object.py').write_text('# recipe')
    cache = tmp_path / 'Object.glb'
    cache.write_bytes(b'previous asset')
    new = b'glTF' + b'\0' * 32
    def run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(new)
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(build.subprocess, 'run', run)
    assert build.run_object_py(tmp_path, 'Object', True, cache, tmp_path, 'blender', True)
    assert cache.read_bytes() == new
