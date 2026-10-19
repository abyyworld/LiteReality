#!/usr/bin/env python3
"""Shrink a RoomPlan scan folder. Never modifies the input.

A capture out of the scanner app is ~63 MB for 59 frames, and most of that is
avoidable:

  * ``metadata.json`` (16.4 MB) is scanner output that nothing in this pipeline
    reads -- no .py or .sh references it. 8 MB of it is ``cameraSamples``, the
    continuous ARKit pose track; the per-frame poses actually used live in
    ``frame_*.json`` (``cameraPoseARFrame`` / ``intrinsics``).
  * the RGB frames are written at a high JPEG quality (~730 KB each at
    1920x1440) and recompress well.
  * ``pointcloud.pcd`` is ``DATA ascii``, which costs ~2.6x its binary size.

Defaults do only the two safe things: drop ``metadata.json`` and recompress the
JPEGs. Resolution, frame count, and the lossless extras are opt-in, because
each has a caveat worth reading (see ``--help`` and the notes below).

    python scripts/capture/shrink_scan.py <scan> --dry-run
    python scripts/capture/shrink_scan.py <scan> --quality 80
    python scripts/capture/shrink_scan.py <scan> --quality 80 --long-edge 1024 --extras

RESOLUTION CAVEAT. The ``intrinsics`` in every ``frame_*.json`` are written for
1920x1440, and ``scene_init/object_init/extract/extract_scene.py:99`` hardcodes that
pair. That call site reads the *depth* image, so downscaling the RGB does not
break it -- but any code that maps RGB pixels to 3D with those intrinsics does
need to derive its own scale factor. ``--long-edge`` leaves ``frame_*.json``
untouched so the stored intrinsics stay 1920-referenced and correct.
"""

from __future__ import annotations

import argparse
import io
import shutil
import struct
import sys
from pathlib import Path

from PIL import Image

FRAME_GLOB = "frame_*.jpg"
DROP_NAMES = {"metadata.json", ".DS_Store"}


def human(n: float) -> str:
    return f"{n / 1e6:.2f} MB"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def frame_index(path: Path) -> int:
    """`frame_00042.jpg` -> 42."""
    return int(path.stem.split("_")[1])


def selected_frames(scan: Path, keep: int | None) -> set[int] | None:
    """Evenly spaced subset of frame indices, or None for all.

    Filenames keep their original numbering: everything downstream globs
    `frame_*`, so gaps are harmless and renumbering would silently break any
    cross-reference to a frame index.
    """
    frames = sorted(scan.glob(FRAME_GLOB), key=frame_index)
    if keep is None or keep >= len(frames):
        return None
    step = len(frames) / keep
    return {frame_index(frames[min(int(i * step), len(frames) - 1)]) for i in range(keep)}


def companions(index: int) -> set[str]:
    """Files that belong to one frame and must travel with it."""
    return {
        f"frame_{index:05d}.jpg",
        f"frame_{index:05d}.json",
        f"depth_{index:05d}.png",
        f"conf_{index:05d}.png",
    }


ENCODERS = {
    "jpeg": lambda q: ("JPEG", dict(quality=q, optimize=True, progressive=True)),
    "webp": lambda q: ("WEBP", dict(quality=q, method=6)),
    "avif": lambda q: ("AVIF", dict(quality=q)),
}


def recompress(src: Path, fmt: str, quality: int, long_edge: int | None) -> bytes:
    """Re-encode one frame. Resolution is preserved unless --long-edge is given.

    At matched quality (measured by PSNR against the source) WebP is ~28% smaller
    than JPEG and AVIF ~40%, at identical pixel dimensions -- so the cheapest way
    to shrink a scan without touching resolution is to change codec, not quality.
    """
    im = Image.open(src)
    if long_edge and max(im.size) > long_edge:
        ratio = long_edge / max(im.size)
        im = im.resize((round(im.width * ratio), round(im.height * ratio)), Image.LANCZOS)
    pil_fmt, kwargs = ENCODERS[fmt](quality)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, pil_fmt, **kwargs)
    return buf.getvalue()


def reoptimize_png(src: Path) -> bytes:
    """Re-encode a PNG at max compression. Pixels are bit-identical."""
    buf = io.BytesIO()
    Image.open(src).save(buf, "PNG", optimize=True, compress_level=9)
    return buf.getvalue()


def pcd_ascii_to_binary(src: Path) -> bytes | None:
    """Convert a `DATA ascii` PCD to `DATA binary`. None if the layout is not
    the plain float32 one the scanner writes -- better to copy it untouched
    than to emit a subtly malformed cloud."""
    raw = src.read_bytes()
    try:
        head, body = raw.split(b"DATA ascii\n", 1)
    except ValueError:
        return None
    fields: dict[str, list[str]] = {}
    for line in head.decode("ascii", "replace").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, rest = line.partition(" ")
        fields[key] = rest.split()
    if (
        fields.get("FIELDS") != ["x", "y", "z"]
        or fields.get("SIZE") != ["4", "4", "4"]
        or fields.get("TYPE") != ["F", "F", "F"]
        or fields.get("COUNT") != ["1", "1", "1"]
    ):
        return None

    out = bytearray()
    count = 0
    for line in body.split(b"\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            return None
        out += struct.pack("<fff", *(float(p) for p in parts))
        count += 1
    if count != int(fields["POINTS"][0]):
        return None
    return head + b"DATA binary\n" + bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scan", type=Path, help="Scan folder to shrink (never modified).")
    ap.add_argument(
        "--out",
        type=Path,
        help="Destination folder. Default: <scan>_slim beside the input.",
    )
    ap.add_argument("--quality", type=int, default=80, help="Encoder quality (default 80).")
    ap.add_argument(
        "--format",
        choices=sorted(ENCODERS),
        default="jpeg",
        help="Frame codec. webp/avif are much smaller at equal PSNR (default jpeg).",
    )
    ap.add_argument(
        "--keep-jpg-extension",
        action="store_true",
        help="With --format webp/avif, still name files frame_*.jpg. PIL and cv2 both "
        "sniff magic bytes so this works, but a .jpg that is not a JPEG is a trap for "
        "the next reader -- prefer renaming the globs.",
    )
    ap.add_argument(
        "--long-edge",
        type=int,
        help="Downscale frames so the long edge is at most this. See RESOLUTION CAVEAT.",
    )
    ap.add_argument(
        "--frames",
        type=int,
        help="Keep only N evenly spaced frames (with their depth/conf/json).",
    )
    ap.add_argument(
        "--extras",
        action="store_true",
        help="Also do the lossless wins: PCD ascii->binary, PNG re-optimize.",
    )
    ap.add_argument(
        "--keep-metadata", action="store_true", help="Keep metadata.json instead of dropping it."
    )
    ap.add_argument("--dry-run", action="store_true", help="Report projected sizes, write nothing.")
    args = ap.parse_args()

    scan: Path = args.scan.expanduser().resolve()
    if not scan.is_dir():
        print(f"not a directory: {scan}", file=sys.stderr)
        return 1
    out: Path = (args.out or scan.with_name(scan.name + "_slim")).expanduser().resolve()
    if out.exists() and not args.dry_run:
        print(f"destination already exists: {out}", file=sys.stderr)
        return 1

    keep = selected_frames(scan, args.frames)
    keep_names: set[str] | None = None
    if keep is not None:
        keep_names = set().union(*(companions(i) for i in keep))

    before = after = 0
    by_group: dict[str, list[int]] = {}

    def note(group: str, src_bytes: int, dst_bytes: int) -> None:
        nonlocal before, after
        before += src_bytes
        after += dst_bytes
        acc = by_group.setdefault(group, [0, 0])
        acc[0] += src_bytes
        acc[1] += dst_bytes

    for src in sorted(scan.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(scan)
        size = src.stat().st_size

        if src.name in DROP_NAMES and not (src.name == "metadata.json" and args.keep_metadata):
            note(f"dropped {src.name}", size, 0)
            continue
        if keep_names is not None and src.name.startswith(("frame_", "depth_", "conf_")):
            if src.name not in keep_names:
                note("dropped frames", size, 0)
                continue

        payload: bytes | None = None
        group = "copied"
        if src.match(FRAME_GLOB):
            payload = recompress(src, args.format, args.quality, args.long_edge)
            group = f"frames ({args.format} q{args.quality})"
            if args.format != "jpeg" and not args.keep_jpg_extension:
                rel = rel.with_suffix("." + args.format)
        elif args.extras and src.suffix == ".png":
            payload = reoptimize_png(src)
            group = "depth/conf (re-optimized)"
        elif args.extras and src.suffix == ".pcd":
            payload = pcd_ascii_to_binary(src)
            group = "pointcloud (binary)" if payload else "pointcloud (copied, unknown layout)"

        note(group, size, len(payload) if payload is not None else size)

        if args.dry_run:
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if payload is None:
            shutil.copy2(src, dst)
        else:
            dst.write_bytes(payload)

    print(f"{scan.name}  ->  {out.name}{'  (dry run)' if args.dry_run else ''}\n")
    print(f"  {'group':<34}{'before':>12}{'after':>12}")
    for group, (b, a) in sorted(by_group.items(), key=lambda kv: -kv[1][0]):
        print(f"  {group:<34}{human(b):>12}{human(a):>12}")
    saved = (1 - after / before) * 100 if before else 0
    print(f"  {'':<34}{'':>12}{'':>12}")
    print(f"  {'TOTAL':<34}{human(before):>12}{human(after):>12}   ({saved:.0f}% smaller)")

    if not args.dry_run:
        actual = dir_size(out)
        print(f"\n  wrote {out}  ({human(actual)} on disk)")
    if args.long_edge:
        print("\n  note: frames were downscaled; frame_*.json intrinsics are left at their")
        print("        1920x1440 reference on purpose. See RESOLUTION CAVEAT in this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
