"""Teach gsplat 1.5.3's `examples/datasets/colmap.py` an explicit test list, at image build.

    python patch_colmap_split.py /opt/gsplat/examples/datasets/colmap.py

Two edits, both inert unless their input is there, so any other dataset trains as gsplat does:

1. Split. When `<data_dir>/test_images.txt` exists (one image name a line), the val set is exactly
   those images and the train set every other image, instead of gsplat's every-`test_every`-th
   split. Eyeful Tower's official v2 split holds out all of camera 17 (scripts/eyeful/
   prepare_eyeful.py writes the list from splits.json). Every listed name must be in the model.
2. Common training crop. With env `LR_TRAIN_CROP=1`, each *training* image is centre-cropped to the
   smallest width and height over the training cameras (K's principal point shifted to match), so
   a batch of several photos from cameras of slightly different sizes stacks (Eyeful's 22 PINHOLE
   cameras are 1471-1486 x 2232-2260 px). Test images are never cropped.

Each anchor must match exactly once, or the image build fails here.

`--eval-jpeg <simple_trainer.py>` instead makes the trainer's eval write its ground truth | render
canvases as JPEG q92 rather than PNG (a 2950 x 2250 PNG takes ~2 s to compress, ~150 s of H100
for 71 test images). The metrics are computed on the float tensors, never on the written files.
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "# LiteReality explicit split (scripts/eyeful/patch_colmap_split.py)"

EDITS: list[tuple[str, str]] = [
    (
        "        indices = np.arange(len(self.parser.image_names))\n"
        "        if split == \"train\":\n"
        "            self.indices = indices[indices % self.parser.test_every != 0]\n"
        "        else:\n"
        "            self.indices = indices[indices % self.parser.test_every == 0]\n",
        "        indices = np.arange(len(self.parser.image_names))\n"
        f"        {MARKER}\n"
        "        _test_file = os.path.join(self.parser.data_dir, \"test_images.txt\")\n"
        "        if os.path.exists(_test_file):\n"
        "            with open(_test_file) as _f:\n"
        "                _test = {l.strip() for l in _f if l.strip()}\n"
        "            _is_test = np.array([n in _test for n in self.parser.image_names])\n"
        "            assert _is_test.sum() == len(_test), (\n"
        "                f\"test_images.txt: {len(_test)} names, {_is_test.sum()} in the model\")\n"
        "            self.indices = indices[~_is_test] if split == \"train\" else indices[_is_test]\n"
        "            print(f\"[Dataset] {split}: {len(self.indices)} images (test_images.txt)\")\n"
        "        elif split == \"train\":\n"
        "            self.indices = indices[indices % self.parser.test_every != 0]\n"
        "        else:\n"
        "            self.indices = indices[indices % self.parser.test_every == 0]\n"
        "        self.crop = None\n"
        "        if split == \"train\" and os.environ.get(\"LR_TRAIN_CROP\") == \"1\":\n"
        "            _sizes = [self.parser.imsize_dict[self.parser.camera_ids[i]]\n"
        "                      for i in self.indices]\n"
        "            self.crop = (min(s[0] for s in _sizes), min(s[1] for s in _sizes))\n"
        "            print(f\"[Dataset] train images centre-cropped to {self.crop}\")\n",
    ),
    (
        "        if self.patch_size is not None:\n"
        "            # Random crop.\n",
        f"        {MARKER}\n"
        "        if self.crop is not None:\n"
        "            _h, _w = image.shape[:2]\n"
        "            _x0, _y0 = (_w - self.crop[0]) // 2, (_h - self.crop[1]) // 2\n"
        "            image = image[_y0 : _y0 + self.crop[1], _x0 : _x0 + self.crop[0]]\n"
        "            K[0, 2] -= _x0\n"
        "            K[1, 2] -= _y0\n"
        "        if self.patch_size is not None:\n"
        "            # Random crop.\n",
    ),
]


def patch(source: str) -> str:
    if MARKER in source:
        raise ValueError("colmap.py is already patched")
    for anchor, replacement in EDITS:
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"anchor found {count} times (want 1): {anchor.strip()[:80]!r}")
        source = source.replace(anchor, replacement)
    if "\nimport os\n" not in source:
        raise ValueError("colmap.py does not import os")
    compile(source, "colmap.py", "exec")
    return source


JPEG_EDITS: list[tuple[str, str]] = [
    ('                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",\n'
     "                    canvas,\n",
     '                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.jpg",\n'
     "                    canvas,\n"
     "                    quality=92,  # LiteReality: JPEG, not PNG (patch_colmap_split.py)\n"),
]


def apply(source: str, edits: list[tuple[str, str]], name: str) -> str:
    for anchor, replacement in edits:
        count = source.count(anchor)
        if count != 1:
            raise ValueError(f"{name}: anchor found {count} times (want 1): {anchor.strip()[:80]!r}")
        source = source.replace(anchor, replacement)
    compile(source, name, "exec")
    return source


if __name__ == "__main__":
    if sys.argv[1] == "--eval-jpeg":
        path = Path(sys.argv[2])
        path.write_text(apply(path.read_text(), JPEG_EDITS, "simple_trainer.py"))
        print(f"patched {path} (eval renders as JPEG)")
    else:
        path = Path(sys.argv[1])
        path.write_text(patch(path.read_text()))
        print(f"patched {path} ({len(EDITS)} edits)")
