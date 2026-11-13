"""The physics study's first two columns: what an LLM guesses from a photo, beside what
LiteReality estimates, for every object of a scanned room — and empty columns for what you measure.

    uv run python scripts/physics_study/llm_estimates.py run/tea_room [--model gpt-5-mini]

Reads ``run/<scan>/objects/quick.json`` (LiteReality's per-body category, size, mass, friction,
restitution and material guess: the same density table every stage uses) and each object's own
masked photo from the realistic stage (``run/<scan>/objects/crops/<name>.png``). Each photo goes
to an OpenAI vision model with the object's category and scanned size, which answers in a fixed
JSON schema (mass, friction, restitution, material, each with a low-high range). Writes
``results/<scan>/physics_study.csv`` and ``.md``. Needs OPENAI_API_KEY in .env (never printed).

The measured columns are left for the bench protocol (``results/<scan>/physics_protocol.md``):
mass on a kitchen scale, static friction by the tilt angle at which it starts to slide (mu =
tan(angle)) on the room's own floor, restitution from drop height and first bounce height
(e = sqrt(h_bounce / h_drop)).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "gpt-5-mini"
API = "https://api.openai.com/v1/responses"

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "material": {"type": "string"},
        "mass_kg": {"type": "number"}, "mass_low_kg": {"type": "number"},
        "mass_high_kg": {"type": "number"},
        "static_friction": {"type": "number"}, "friction_low": {"type": "number"},
        "friction_high": {"type": "number"},
        "restitution": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": ["material", "mass_kg", "mass_low_kg", "mass_high_kg", "static_friction",
                 "friction_low", "friction_high", "restitution", "reasoning"],
}

PROMPT = """You are estimating real physical properties of one object from a scanned room, for a \
physics simulation. The photo shows the object cut out of the scan. Its category from the scanner \
is "{category}" and its measured bounding box is {w:.2f} m x {d:.2f} m x {h:.2f} m (width x depth \
x height).

Estimate, for this exact object as it appears:
- mass_kg (with a plausible low-high range),
- the static friction coefficient of the object resting on the floor of a typical office or home \
(with a range),
- the coefficient of restitution if dropped onto that floor,
- its main material.
Answer from the photo and the size; if unsure, still give your best estimate with a wide range. \
Keep the reasoning to one or two sentences."""


def _key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        env = Path(".env")
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (in the environment or .env)")
    return key


def _photo(path: Path) -> str:
    """The masked crop on white (its transparent parts would read as black), as a data URL."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGBA")
        im.thumbnail((768, 768))
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def ask(key: str, model: str, category: str, size: list[float], photo: Path | None) -> dict:
    w, d, h = (float(v) for v in size)
    content = [{"type": "input_text",
                "text": PROMPT.format(category=category, w=w, d=d, h=h)}]
    if photo is not None:
        content.append({"type": "input_image", "image_url": _photo(photo)})
    body = {"model": model, "input": [{"role": "user", "content": content}],
            "text": {"format": {"type": "json_schema", "name": "physics", "schema": SCHEMA,
                                "strict": True}}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)
            break
        except urllib.error.HTTPError as exc:
            detail = json.loads(exc.read() or b"{}").get("error", {})
            if exc.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise SystemExit(f"OpenAI {exc.code}: {detail.get('code')} "
                             f"{(detail.get('message') or '')[:200].split('sk-')[0]}") from exc
    text = "".join(c.get("text", "") for o in out.get("output", [])
                   for c in (o.get("content") or []) if c.get("type") == "output_text")
    answer = json.loads(text)
    answer["tokens"] = (out.get("usage") or {}).get("total_tokens")
    return answer


COLUMNS = ["object", "category", "size_m", "photo",
           "lite_material", "lite_mass_kg", "lite_friction", "lite_restitution",
           "llm_material", "llm_mass_kg", "llm_mass_range_kg", "llm_friction",
           "llm_friction_range", "llm_restitution", "llm_reasoning",
           "measured_mass_kg", "measured_tilt_deg", "measured_friction",
           "measured_drop_m", "measured_bounce_m", "measured_restitution", "notes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="run/<scan>")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: results/<scan>/physics_study.csv")
    args = parser.parse_args()
    run = args.run_dir
    quick = json.loads((run / "objects" / "quick.json").read_text())
    scan = run.name
    out = args.out or Path("results") / scan / "physics_study.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    key = _key()
    rows, tokens = [], 0
    for name, body in sorted(quick["bodies"].items()):
        photo = run / "objects" / "crops" / f"{name}.png"
        photo = photo if photo.is_file() else None
        t = time.perf_counter()
        a = ask(key, args.model, body.get("category", "object"), body.get("size", [0, 0, 0]),
                photo)
        tokens += a.get("tokens") or 0
        print(f"  {name:16s} LiteReality {body.get('mass_kg', 0):7.2f} kg mu "
              f"{body.get('friction', 0):.2f} | {args.model} {a['mass_kg']:7.2f} kg "
              f"({a['mass_low_kg']:.1f}-{a['mass_high_kg']:.1f}) mu {a['static_friction']:.2f} "
              f"{a['material']}  [{time.perf_counter() - t:.1f} s]", flush=True)
        rows.append({
            "object": name, "category": body.get("category"),
            "size_m": " x ".join(f"{v:.2f}" for v in body.get("size", [])),
            "photo": "yes" if photo else "no (size only)",
            "lite_material": body.get("material_guess"), "lite_mass_kg": body.get("mass_kg"),
            "lite_friction": body.get("friction"), "lite_restitution": body.get("restitution"),
            "llm_material": a["material"], "llm_mass_kg": round(a["mass_kg"], 2),
            "llm_mass_range_kg": f"{a['mass_low_kg']:.1f}-{a['mass_high_kg']:.1f}",
            "llm_friction": round(a["static_friction"], 2),
            "llm_friction_range": f"{a['friction_low']:.2f}-{a['friction_high']:.2f}",
            "llm_restitution": round(a["restitution"], 2), "llm_reasoning": a["reasoning"],
        })
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    md = [f"# {scan}: physics estimates — LiteReality vs {args.model} (measured: to fill in)", "",
          "| Object | Category | Size (m) | LiteReality kg | LLM kg (range) | LiteReality mu | "
          "LLM mu (range) | LLM e | LLM material |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['object']} | {r['category']} | {r['size_m']} | {r['lite_mass_kg']} | "
                  f"{r['llm_mass_kg']} ({r['llm_mass_range_kg']}) | {r['lite_friction']} | "
                  f"{r['llm_friction']} ({r['llm_friction_range']}) | {r['llm_restitution']} | "
                  f"{r['llm_material']} |")
    md += ["", f"Model {args.model}, {tokens} tokens in all, "
               f"{time.strftime('%Y-%m-%d')}. Each object was sent its masked photo from the "
               "realistic stage (where it has one), its scanner category and its scanned size.",
           "", "Measured columns are in the CSV; fill them with the protocol in "
               "`physics_protocol.md`."]
    out.with_suffix(".md").write_text("\n".join(md) + "\n")
    print(f"wrote {out} and {out.with_suffix('.md')} ({tokens} tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
