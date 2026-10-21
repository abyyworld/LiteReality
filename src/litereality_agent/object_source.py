"""Is an object's `object.py` a complete definition of it?

`object.py` is the editable source of every recipe-built object: `blender -b --python object.py --
<texdir> <out.glb>` has to rebuild the object's GLB on any machine. The pipeline used to accept one
on the strength of `is_file()` alone, and on tea_room an agent-built door shipped an `object.py`
that ended at the `OBJECT DEFINITION` marker — the inlined helpers and nothing else. It is valid
Python, so Blender ran it, exited 0, and wrote no GLB; the seed bake then "regenerated" the door
into nothing and `Room.glb` never assembled, with no error naming the file.

This module says what is wrong with an `object.py` in words a person can act on, without running
Blender. It lives at the package top level, outside the layers, because the producer
(`models/object_generation`), the progress board (`pipeline/scene_init`) and the room compiler
(`room_ops`) all have to apply the SAME test — the bar, the builder and the bake have to agree on
what "built" means. `make_selfcontained.py` carries its own copy of the check because it runs as a
standalone script inside the agent's skill directory, where this package may not be importable.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

# Written by make_selfcontained.py between the inlined helper library and the recipe's own code.
EDIT_HEADER = "# ==== OBJECT DEFINITION — edit below to change the object ===="


def _calls_export(tree: ast.AST) -> bool:
    """A call that writes the GLB: blender_lib's `export_glb(...)` (bare or as `bl.export_glb`), or
    the raw `bpy.ops.export_scene.gltf(...)` operator a static wrapper uses. `import_scene.gltf`
    reads a GLB and does not count."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "export_glb":
            return True
        if isinstance(func, ast.Attribute):
            if func.attr == "export_glb":
                return True
            if (func.attr == "gltf" and isinstance(func.value, ast.Attribute)
                    and func.value.attr == "export_scene"):
                return True
    return False


def problems_in_text(text: str) -> list[str]:
    """What stops this source from rebuilding its object. Empty means it looks complete."""
    if not text.strip():
        return ["object.py is empty"]
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [f"object.py does not parse (line {exc.lineno}: {exc.msg}) — truncated?"]

    region = tree
    if EDIT_HEADER in text:
        # A bundled file: everything above the marker is the inlined helper library, which
        # DEFINES `export_glb` (and calls the glTF operator inside it). Only the section below the
        # marker builds anything, so that is where the build body and its export have to be.
        section = text.split(EDIT_HEADER, 1)[1]
        try:
            region = ast.parse(section)
        except SyntaxError as exc:
            return [f"the OBJECT DEFINITION section does not parse (line {exc.lineno}: {exc.msg})"]
        if not region.body:
            return ["the OBJECT DEFINITION section is empty — the build body and its export_glb() "
                    "call are missing (a truncated bundle)"]
    if not _calls_export(region):
        return ["object.py never exports a GLB (no export_glb() or export_scene.gltf() call)"]
    return []


@lru_cache(maxsize=256)
def _problems_cached(path: str, _mtime_ns: int, _size: int) -> tuple[str, ...]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return (f"object.py is unreadable ({type(exc).__name__}: {exc})",)
    return tuple(problems_in_text(text))


def problems(path: Path | str) -> list[str]:
    """`problems_in_text` for a file, cached on its mtime and size (the build board polls it)."""
    path = Path(path)
    try:
        stat = path.stat()
    except OSError:
        return ["no object.py"]
    return list(_problems_cached(str(path), stat.st_mtime_ns, stat.st_size))


def is_complete(path: Path | str) -> bool:
    return not problems(path)
