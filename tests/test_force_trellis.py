"""$LR_FORCE_TRELLIS — sending one object to TRELLIS in a way re-classification cannot undo."""

from __future__ import annotations

from litereality_agent.pipeline.scene_init.reconstruct.classify import classify_complexity as cc
from litereality_agent.settings import LiteRealitySettings

PROCEDURAL = {"route": "procedural", "is_organic": False}


def test_a_named_object_is_forced_to_trellis_and_says_why(monkeypatch):
    monkeypatch.setenv("LR_FORCE_TRELLIS", "Table0, dishwasher0")
    decision = cc.apply_policy("objects", "dishwasher", PROCEDURAL, name="Dishwasher0")
    assert decision["route"] == "trellis"
    assert decision["route_vlm"] == "procedural"
    assert decision["forced_by_policy"] == "LR_FORCE_TRELLIS"


def test_other_objects_keep_their_route(monkeypatch):
    monkeypatch.setenv("LR_FORCE_TRELLIS", "Dishwasher0")
    assert cc.apply_policy("objects", "cabinet", PROCEDURAL, name="Cabinet0")["route"] == "procedural"
    monkeypatch.delenv("LR_FORCE_TRELLIS")
    assert cc.apply_policy("objects", "dishwasher", PROCEDURAL, name="Dishwasher0")["route"] == "procedural"


def test_the_setting_reaches_the_classify_subprocess():
    settings = LiteRealitySettings(_env_file=None, LR_FORCE_TRELLIS="Dishwasher0")
    assert settings.as_environment()["LR_FORCE_TRELLIS"] == "Dishwasher0"
