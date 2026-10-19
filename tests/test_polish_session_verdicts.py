"""Optional polish sessions must report provider failures to their parent stage."""
import asyncio

import pytest

from litereality_agent.agent import providers
from litereality_agent.pipeline.realism_authoring.author import materials, quality


@pytest.mark.parametrize('module', [materials, quality])
@pytest.mark.parametrize('terminal, expected', [
    (providers.SessionResult(result='done'), 0),
    (providers.SessionResult(result='auth failed', is_error=True), 1),
    (providers.SessionResult(stopped='step budget'), 1),
    (None, 1),
])
def test_polish_requires_clean_terminal_result(tmp_path, monkeypatch, module, terminal, expected):
    room = tmp_path / 'scan' / 'realism_authoring' / 'room'
    room.mkdir(parents=True)
    (room / 'Room.py').write_text('SHELL = {}')
    refs = tmp_path / 'refs'
    refs.mkdir()
    class Harness:
        name = 'codex'
        supports = frozenset()
        def effective_model(self, spec):
            return 'gpt-6-astra'
        async def run(self, spec):
            if terminal is not None:
                yield terminal
    monkeypatch.setattr(providers, 'resolve', lambda *args: Harness())
    monkeypatch.setattr(module, 'surfaces_for', lambda *args: [])
    monkeypatch.setattr('litereality_agent.agent.scratch.bind', lambda **kwargs: None)
    if module is materials:
        monkeypatch.setattr(module, '_flat_sites', lambda *args: ('', 0))
        monkeypatch.setattr(module, '_fixture_keys', lambda *args: ('', 0, 0))
        monkeypatch.setattr(module, '_object_lines', lambda *args: '')
    else:
        monkeypatch.setattr(module, '_object_lines', lambda *args: ('', 0))
    assert asyncio.run(module.run(room, refs, refs, refs, 'test', 1)) == expected
