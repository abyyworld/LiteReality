"""A cached asset must not hide failure of its edited Blender recipe."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from litereality_agent.room_ops.compile import build_from_room as build


@pytest.mark.parametrize('returncode, output', [(0, None), (1, b'glTF' + b'\0' * 32), (0, b'bad')])
def test_failed_regeneration_preserves_old_cache_but_returns_failure(tmp_path, monkeypatch, returncode, output):
    # Statically complete (it calls its export), so the failure is Blender's to report.
    (tmp_path / 'object.py').write_text('raise RuntimeError("broken recipe")\nexport_glb(OUT_GLB)\n')
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
    (tmp_path / 'object.py').write_text('# recipe\nexport_glb(OUT_GLB)\n')
    cache = tmp_path / 'Object.glb'
    cache.write_bytes(b'previous asset')
    new = b'glTF' + b'\0' * 32
    def run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(new)
        return SimpleNamespace(returncode=0, stderr=b'')
    monkeypatch.setattr(build.subprocess, 'run', run)
    assert build.run_object_py(tmp_path, 'Object', True, cache, tmp_path, 'blender', True)
    assert cache.read_bytes() == new


def test_an_object_py_that_builds_nothing_fails_before_blender_runs(tmp_path, monkeypatch, capsys):
    """tea_room's door: the inlined helpers and the OBJECT DEFINITION marker, nothing after."""
    from litereality_agent.object_source import EDIT_HEADER

    (tmp_path / 'object.py').write_text(f'def export_glb(p):\n    pass\n\n{EDIT_HEADER}\n')
    cache = tmp_path / 'Object.glb'
    cache.write_bytes(b'previous asset')

    def run(cmd, **kwargs):
        raise AssertionError('Blender must not be started for a source that cannot build')

    monkeypatch.setattr(build.subprocess, 'run', run)
    assert not build.run_object_py(tmp_path, 'Object', True, cache, tmp_path, 'blender', True)
    assert cache.read_bytes() == b'previous asset'
    assert 'OBJECT DEFINITION section is empty' in capsys.readouterr().out
