"""Epic RealityScan 2.2 (Linux build: the Windows app in a bundled CrossOver Wine) on Modal: camera
alignment (SfM) for a research comparison of its poses against ARKit's. INTERNAL RESEARCH ONLY: its
EULA forbids offering it to third parties as a hosted service, so it never joins the served
pipeline, the room call or any public endpoint. Nothing here is deployed; run it with `modal run`.

The package is NOT public (Epic account download). It sits on the private Volume
`litereality-tools` at /realityscan/RealityScan-2.2.deb and is never copied anywhere public.

    export MODAL_IMAGE_BUILDER_VERSION=2025.06
    uv run --env-file .env modal run deploy/modal/realityscan/app.py::deb_info   # control file
    uv run --env-file .env modal run deploy/modal/realityscan/app.py::smoke      # 10 images
"""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import modal

APP_NAME = "litereality-realityscan"
VOLUME = "litereality-tools"
VOL = Path("/tools")
DEB = VOL / "realityscan" / "RealityScan-2.2.deb"
DEB_SHA256 = "b0778e21768cffee57aae20cf33679399b408356c04ebd66c51f51e13614eaaa"
RS = Path("/opt/realityscan/bin")

vol = modal.Volume.from_name(VOLUME)
app = modal.App(APP_NAME)

base = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .run_commands("apt-get update && apt-get install -y --no-install-recommends file binutils"
                  " ca-certificates && rm -rf /var/lib/apt/lists/*")
)


def _sh(cmd: str, timeout: int = 600, env: dict | None = None) -> dict:
    t = time.perf_counter()
    try:
        p = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        code, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") + (exc.stderr or b"")
        code, out = "timeout", out.decode(errors="replace") if isinstance(out, bytes) else str(out)
    res = {"cmd": cmd, "code": code, "s": round(time.perf_counter() - t, 1), "out": out[-6000:]}
    print(f"$ {cmd}\n[{code}] {res['out'][-3000:]}", flush=True)
    return res


@app.function(image=base, volumes={VOL: vol}, cpu=1.0, memory=2048, timeout=900)
def deb_info() -> list[dict]:
    """The .deb's control file (Depends) and what kind of binaries its Wine is."""
    return [
        _sh(f"ls -la {DEB}; sha256sum {DEB}"),
        _sh(f"dpkg-deb -I {DEB}"),
        _sh(f"mkdir -p /x && dpkg-deb -x {DEB} /x && ls -la /x/opt/realityscan /x/opt/realityscan/bin"
            " && file /x/opt/realityscan/bin/* | head -40", timeout=900),
        _sh("cd /x/opt/realityscan && ls lib lib64 2>/dev/null | head -40; find . -maxdepth 4 -name"
            " 'wine*' -type f | head -30 | xargs file | head -30; du -sh /x/opt/realityscan;"
            " head -c 4000 bin/wine | strings | head -80"),
    ]


# --- the RealityScan image -------------------------------------------------------------------
# Ubuntu 24.04 + the .deb's Depends (apt resolves them from the .deb itself) + Xvfb and a Vulkan
# loader. The .deb is installed at image build with the private Volume mounted, so the image (private
# to this Modal workspace, never published) holds /opt/realityscan; the .deb file itself is not kept.
def _install_deb() -> None:
    import hashlib
    h = hashlib.sha256()
    with open(DEB, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == DEB_SHA256, h.hexdigest()
    # Its Depends from apt, then the package unpacked WITHOUT its postinst: that runs CrossOver's
    # `cxupgrade --install-bottles`, which hangs at image build (and >10 min in a container). It is
    # not needed: the first `wine --bottle realityscan` copies the bottle template from
    # /opt/realityscan/support/realityscan to /root/.realityscan/realityscan (~1 min).
    subprocess.run(["bash", "-lc", f"dpkg --unpack {DEB}; I=/var/lib/dpkg/info; mv $I/realityscan.postinst"
                    " $I/realityscan.postinst.orig && apt-get update -qq && apt-get -f install -y -qq"
                    " --no-install-recommends && rm -rf /var/lib/apt/lists/*"], check=True)
    subprocess.run(["bash", "-lc", "dpkg -s realityscan | grep -q 'install ok installed'"], check=True)


rs_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .env({"DEBIAN_FRONTEND": "noninteractive", "NVIDIA_VISIBLE_DEVICES": "all",
          "NVIDIA_DRIVER_CAPABILITIES": "all", "XDG_RUNTIME_DIR": "/tmp/runtime-root"})
    .run_commands("apt-get update && apt-get install -y --no-install-recommends xvfb xauth"
                  " x11-utils libvulkan1 mesa-vulkan-drivers vulkan-tools libgl1 libegl1 winbind"
                  " file procps psmisc ca-certificates && rm -rf /var/lib/apt/lists/*")
    .run_function(_install_deb, volumes={VOL: vol}, cpu=2.0, memory=4096, timeout=1800)
    .apt_install("binutils")
)

XVFB = "Xvfb :99 -screen 0 1600x1200x24 -nolisten tcp >/tmp/xvfb.log 2>&1 &"
RS_ENV = {"DISPLAY": ":99", "WINEDEBUG": "-all", "XDG_RUNTIME_DIR": "/tmp/runtime-root"}
WINE = f"{RS}/wine --bottle realityscan --wait-children"
# "Skip for now" at the first-run Epic sign-in, no login (github.com/kersplody/realityscan-gs, MIT).
SKIP_LOGIN = (f"{WINE} reg.exe add 'HKCU\\Software\\EpicGames.RealityScan\\RealityScan\\Workspace'"
              " /v eosLoginType /t REG_DWORD /d 2 /f")
RS_DIR = "C:/Program Files/Epic Games/RealityScan"
BOTTLE = Path("/root/.realityscan/realityscan")
RS_LOG = BOTTLE / "drive_c/users/crossover/AppData/Local/Temp/RealityScan.log"

# The Export Registration params.xml (RealityCapture's format: the Configuration id is that
# dialog's; keys found in RealityScan.exe's strings and tried one by one on 2.2.0.119430):
#   calexFileFormat=COLMAP        the COLMAP exporter (sparse/0/{cameras,images,points3D})
#   calexExportUndistorted=0      the cameras as aligned: FULL_OPENCV on the ORIGINAL images
#                                 (default 1: undistorted, cropped PNG copies with PINHOLE)
#   calexExportImages=0           no image copies: images.txt names the input files as given
#                                 (with images, they are renamed 00000.png, 00001.png, ...)
#   colmapFileType=0 text, 1 binary (*.bin); colmapDirStructure=1 flat (no sparse/0)
# Do NOT name the export target "cameras.txt": the exporter then drops cameras.txt.
COLMAP_PARAMS = {"calexFileFormat": "COLMAP", "calexExportUndistorted": "0",
                 "calexExportImages": "0", "colmapFileType": "0"}
PARAMS_ID = "{2D5793BC-A65D-4318-A1B9-A05044608385}"
# Silent mode answers the two export popups itself ("only the selection?" Yes; "FULL_OPENCV ok?"
# Yes); the keys make that explicit. appQuitOnError: a failed command exits instead of hanging.
SETTINGS = ['-set "appQuitOnError=true"', '-set "PUS-1-323292754=1"', '-set "PUS-1-323301212=1"']


def z(p: Path | str) -> str:
    """A Linux path as Wine's Z: drive sees it, single-quoted for the shell. Unquoted, bash eats
    the backslashes and RealityScan crashes at start (0xC000001D in its path handling)."""
    return "'Z:" + str(p).replace("/", "\\") + "'"


def params_xml(entries: dict[str, str]) -> bytes:
    body = "".join(f'<entry key="{k}" value="{v}"/>\n' for k, v in entries.items())
    return f'<Configuration id="{PARAMS_ID}">\n{body}</Configuration>\n'.encode()


def _setup() -> dict:
    """Xvfb on :99, then the sign-in skip (which also creates the Wine bottle on first use)."""
    os.makedirs("/tmp/runtime-root", mode=0o700, exist_ok=True)
    subprocess.run(["bash", "-lc", f"rm -f /tmp/.X99-lock; {XVFB} sleep 2"], check=False)
    return _sh(f"timeout 600 {SKIP_LOGIN}", timeout=660, env=RS_ENV)


def _rs(args: str, timeout: int = 3600) -> dict:
    """RealityScan.exe (CPU alignment) through the bundled Wine, on the Xvfb display; its own log
    (RealityScan.log, reset per call) comes back as "rs_log"."""
    RS_LOG.parent.mkdir(parents=True, exist_ok=True)
    RS_LOG.write_bytes(b"")
    res = _sh(f"cd /tmp && timeout {timeout} {WINE} --workdir '{RS_DIR}' RealityScan.exe"
              f" {args}", timeout=timeout + 60, env=RS_ENV)
    res["rs_log"] = RS_LOG.read_bytes().decode("utf-8", "replace")[-20000:]
    subprocess.run(["bash", "-lc", "for p in $(pgrep -f '^RealityScan[.]exe'); do kill $p; done"])
    return res


def align_cmd(images: Path, out: Path, focal35: float | None = None,
              distortion_model: int = 2, extra: str = "") -> str:
    """The CLI for: new scene, the folder's images in ONE calibration + lens group (one phone
    camera), an approximate 35 mm-equivalent focal prior if given, align, keep the largest
    component, export COLMAP text to out/colmap/sparse/0, save out/scene.rsproj.
    distortion_model: 0 none, 1 division, 2 Brown3, 3 Brown4, 4/5 + tangential."""
    prior = ['-selectAllImages', '-editInputSelection "inpCalibrationGroup=0"',
             '-editInputSelection "inpLensGroup=0"',
             f'-editInputSelection "inpDistortionModel={distortion_model}"']
    if focal35:
        prior += ['-editInputSelection "inpCalibration=1"',
                  f'-editInputSelection "inpFocal={focal35:.4f}"']
    return " ".join(["-headless", "-silent", z(out / "crash"), *SETTINGS, "-newScene",
                     "-addFolder", z(images), *prior, extra, "-align", "-selectMaximalComponent",
                     "-exportRegistration", z(out / "colmap" / "colmap"), z(out / "params.xml"),
                     "-save", z(out / "scene.rsproj"), "-quit"])


@app.function(image=rs_image, cpu=8.0, memory=16384, timeout=3600)
def align(images_tar: bytes, focal35: float | None = None, distortion_model: int = 2,
          extra: str = "", timeout: int = 3000) -> dict:
    """images_tar: images/*.jpg. -> {"colmap": tar.gz of sparse/0 text, "rs_log", "seconds", ...}.
    INTERNAL RESEARCH ONLY (RealityScan EULA): never call this from a served path."""
    return _align(images_tar, focal35, distortion_model, extra, timeout)


@app.function(image=rs_image, cpu=8.0, memory=16384, timeout=3600, volumes={VOL: vol})
def align_to_volume(images_tar: bytes, tag: str, focal35: float | None = None,
                    distortion_model: int = 2, extra: str = "", timeout: int = 3000) -> str:
    """`align`, its result also written to the private Volume at /runs/<tag>/ (colmap.tar.gz,
    rs_log.txt, run.json), so a `modal run --detach` survives this machine's flaky link; fetch
    with `modal volume get litereality-tools runs/<tag> <dir>`."""
    import json
    res = _align(images_tar, focal35, distortion_model, extra, timeout)
    d = VOL / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "colmap.tar.gz").write_bytes(res.pop("colmap"))
    (d / "rs_log.txt").write_text(res.pop("rs_log"))
    (d / "run.json").write_text(json.dumps({**res, "extra": extra}, indent=1))
    vol.commit()
    return str(d)


def _align(images_tar: bytes, focal35: float | None, distortion_model: int, extra: str,
           timeout: int) -> dict:
    setup = _setup()
    work = Path(tempfile.mkdtemp(prefix="rs-"))
    with tarfile.open(fileobj=io.BytesIO(images_tar)) as tf:
        tf.extractall(work)
    out = work / "out"
    (out / "colmap").mkdir(parents=True)
    (out / "params.xml").write_bytes(params_xml(COLMAP_PARAMS))
    n = sum(1 for _ in (work / "images").iterdir())
    t = time.perf_counter()
    run = _rs(align_cmd(work / "images", out, focal35, distortion_model, extra), timeout=timeout)
    seconds = round(time.perf_counter() - t, 1)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(out / "colmap", arcname="colmap")
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    return {"images": n, "seconds": seconds, "code": run["code"], "files": files,
            "rs_log": run["rs_log"], "stdout": run["out"], "setup": setup,
            "colmap": buf.getvalue()}


@app.function(image=rs_image, cpu=8.0, memory=16384, timeout=3600)
def hold(images_tar: bytes, minutes: int = 40) -> None:
    """Debugging: Xvfb + login skip + images at /work/images, then idle so
    `modal container exec <id> -- bash -c ...` can drive RealityScan by hand."""
    _setup()
    with tarfile.open(fileobj=io.BytesIO(images_tar)) as tf:
        tf.extractall("/work")
    print("READY", flush=True)
    time.sleep(minutes * 60)


def _frames_tar(frames: str, n: int, step: int) -> bytes:
    src = sorted(Path(frames).expanduser().glob("frame_*.jpg"))[::step][:n]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in src:
            tf.add(p, arcname=f"images/{p.name}")
    return buf.getvalue()


@app.local_entrypoint()
def run_hold(frames: str = "~/LiteReality-scans/stray_atrium", n: int = 10, step: int = 2,
             minutes: int = 40) -> None:
    hold.remote(_frames_tar(frames, n, step), minutes)


@app.local_entrypoint()
def smoke(frames: str = "~/LiteReality-scans/stray_atrium", n: int = 10, step: int = 2,
          focal35: float = 25.03, distortion_model: int = 2,
          out: str = "results/realityscan/smoke") -> None:
    """10 Stray frames -> `align` -> results/realityscan/smoke/{colmap/, rs_log.txt, smoke.json}.
    focal35 25.03 = ARKit fx 1335.5 px * 36 mm / 1920 px."""
    import json
    res = align.remote(_frames_tar(frames, n, step), focal35, distortion_model)
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(res.pop("colmap"))) as tf:
        tf.extractall(o)
    (o / "rs_log.txt").write_text(res.pop("rs_log"))
    (o / "smoke.json").write_text(json.dumps(res, indent=1))
    print("images", res["images"], "seconds", res["seconds"], "code", res["code"], res["files"])


# For a long handheld walk through a sparsely textured room, the defaults (Medium detector,
# Medium overlap) split the Stray walk into 23 components (largest 60 of 400) and the video into
# 9 (largest 102 of 388): white walls, motion blur. More, weaker features, the pair search for
# LOW overlap (tries more pairs), and a second -align that re-matches across components.
# The values are the settings' names (Ultra/Low): checked to take effect (features per image
# 3250 -> 16937 with Ultra) in the `hold` container.
ROBUST = " ".join([
    '-set "sfmDetectorSensitivity=Ultra"', '-set "sfmImagesOverlap=Low"',
    '-set "sfmMaxFeaturesPerMpx=20000"', '-set "sfmMaxFeaturesPerImage=80000"',
    '-set "sfmPreselectorFeatures=20000"', '-set "sfmMaxFeatureReprojectionError=3"',
    "-align", '-set "sfmForceComponentRematch=true"'])   # align_cmd then aligns again


def _dir_tar(images: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in sorted(images.glob("*.jpg")) + sorted(images.glob("*.xmp")):
            tf.add(p, arcname=f"images/{p.name}")
    return buf.getvalue()


# Pose priors from XMP sidecars (scripts/realityscan/write_priors.py): honoured (the export comes
# out metric, scale 0.99-1.02 to the priors' frame, but turned 90 deg). Loose: positions
# +-10 cm, orientations +-5 deg: several times ARKit's own drift, so the images still decide the
# poses; the priors mainly join what the images alone leave as separate components.
PRIORS = " ".join([
    '-set "sfmEnableCameraPrior=true"', '-set "sfmCameraPriorAccuracyX=0.1"',
    '-set "sfmCameraPriorAccuracyY=0.1"', '-set "sfmCameraPriorAccuracyZ=0.1"',
    '-set "sfmCameraPriorAccuracyYaw=5"', '-set "sfmCameraPriorAccuracyPitch=5"',
    '-set "sfmCameraPriorAccuracyRoll=5"',
    # Components that each hold enough priors are placed in the priors' frame and merged there
    # (without this, the priors scale and place only the largest component: 21 components, the
    # largest 40 of 400 Stray frames). The export is still turned (Z up, local Euclidean).
    '-set "sfmMergeGeoreferencedComponents=true"', '-set "sfmGeoRef=true"'])


@app.local_entrypoint()
def run_align(images: str, out: str, focal35: float = 0.0, distortion_model: int = 2,
              timeout: int = 3000, extra: str = "", robust: bool = False,
              priors: bool = False) -> None:
    """A folder of JPEGs -> `align` -> <out>/{colmap/sparse/0/*.txt, rs_log.txt, run.json}.
    focal35 0: no focal prior (RealityScan self-calibrates; one calibration + lens group)."""
    import json
    src = Path(images).expanduser()
    data = _dir_tar(src)
    print(f"{src}: {len(data) / 1e6:.0f} MB tar", flush=True)
    t = time.perf_counter()
    if robust:
        extra = f"{ROBUST} {extra}"
    if priors:
        extra = f"{PRIORS} {extra}"
    res = align.remote(data, focal35 or None, distortion_model, extra, timeout)
    o = Path(out).expanduser()
    o.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(res.pop("colmap"))) as tf:
        tf.extractall(o)
    (o / "rs_log.txt").write_text(res.pop("rs_log"))
    res.update({"focal35": focal35 or None, "distortion_model": distortion_model, "extra": extra,
                "round_trip_s": round(time.perf_counter() - t, 1), "cpu": 8, "memory_mb": 16384})
    (o / "run.json").write_text(json.dumps(res, indent=1))
    print("images", res["images"], "seconds", res["seconds"], "code", res["code"], res["files"])


@app.function(image=rs_image, cpu=2.0, memory=4096, timeout=900)
def exe_strings(pattern: str = "sfm") -> str:
    """Setting keys in RealityScan.exe's strings (ASCII and UTF-16) matching `pattern`."""
    exe = "/opt/realityscan/support/realityscan/drive_c/Program Files/Epic Games/RealityScan"
    res = _sh(f"cd '{exe}' 2>/dev/null || cd \"$(dirname \"$(find /opt/realityscan -name"
              f" RealityScan.exe | head -1)\")\"; pwd; for f in *.exe *.dll; do"
              f" strings -a \"$f\"; strings -a -el \"$f\"; done 2>/dev/null"
              f" | grep -E '{pattern}' | sort -u | head -400", timeout=800)
    return res["out"]


@app.local_entrypoint()
def keys(pattern: str = "sfm") -> None:
    print(exe_strings.remote(pattern))


@app.local_entrypoint()
def run_align_detached(images: str, tag: str, focal35: float = 0.0, distortion_model: int = 2,
                       timeout: int = 3000, extra: str = "", robust: bool = False,
                       priors: bool = False) -> None:
    """As `run_align`, but the result goes to the Volume (runs/<tag>); use with --detach."""
    src = Path(images).expanduser()
    data = _dir_tar(src)
    if robust:
        extra = f"{ROBUST} {extra}"
    if priors:
        extra = f"{PRIORS} {extra}"
    print(f"{src}: {len(data) / 1e6:.0f} MB tar -> runs/{tag}", flush=True)
    print(align_to_volume.remote(data, tag, focal35 or None, distortion_model, extra, timeout))


# --- dense runs: inputs and results on the private Volume ---------------------------------------
# Dense frame sets (1000-2500 JPEGs, 250-600 MB) are uploaded once with `modal volume put
# litereality-tools <dir> inputs/<name>/images` and read from the Volume here, so a flaky link
# neither re-sends them per run nor kills a run (`modal run --detach`). Each run writes
# runs/<tag>/: colmap.tar.gz (largest component, COLMAP text), xmp.tar.gz (RealityScan's own XMP
# for every image of every component: pose, calibration and xcr:ComponentId), rs_log.txt (full)
# and run.json.
def dense_cmd(images: Path, out: Path, focal35: float | None, distortion_model: int, extra: str,
              post: str) -> str:
    prior = ['-selectAllImages', '-editInputSelection "inpCalibrationGroup=0"',
             '-editInputSelection "inpLensGroup=0"',
             f'-editInputSelection "inpDistortionModel={distortion_model}"']
    if focal35:
        prior += ['-editInputSelection "inpCalibration=1"',
                  f'-editInputSelection "inpFocal={focal35:.4f}"']
    return " ".join(["-headless", "-silent", z(out / "crash"), *SETTINGS, "-newScene",
                     "-addFolder", z(images), *prior, extra, "-align", post, "-exportXMP",
                     "-selectMaximalComponent",
                     "-exportRegistration", z(out / "colmap" / "colmap"), z(out / "params.xml"),
                     "-save", z(out / "scene.rsproj"), "-quit"])


@app.function(image=rs_image, cpu=8.0, memory=16384, timeout=7200, volumes={VOL: vol})
def align_dir(src: str, tag: str, focal35: float | None = None, distortion_model: int = 2,
              extra: str = "", post: str = "", timeout: int = 6600, xmp_src: str = "") -> dict:
    """src: a Volume folder (relative to its root) holding images/*.jpg; xmp_src: optional Volume
    folder of *.xmp priors copied beside the images. INTERNAL RESEARCH ONLY."""
    import json
    import shutil
    _setup()
    vol.reload()
    work = Path(tempfile.mkdtemp(prefix="rs-"))
    images = work / "images"
    t = time.perf_counter()
    shutil.copytree(VOL / src / "images", images)
    if xmp_src:
        for p in (VOL / xmp_src).glob("*.xmp"):
            shutil.copy2(p, images / p.name)
    copy_s = round(time.perf_counter() - t, 1)
    out = work / "out"
    (out / "colmap").mkdir(parents=True)
    (out / "params.xml").write_bytes(params_xml(COLMAP_PARAMS))
    n = sum(1 for _ in images.glob("*.jpg"))
    n_xmp_in = sum(1 for _ in images.glob("*.xmp"))
    for p in images.glob("*.xmp"):      # priors in, RealityScan's own XMP out: keep both
        shutil.copy2(p, work / (p.name + ".prior"))
    t = time.perf_counter()
    cmd = dense_cmd(images, out, focal35, distortion_model, extra, post)
    run = _rs(cmd, timeout=timeout)
    seconds = round(time.perf_counter() - t, 1)
    log = RS_LOG.read_bytes().decode("utf-8", "replace") if RS_LOG.exists() else run["rs_log"]
    d = VOL / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    for name, root, pattern in (("colmap.tar.gz", out, "colmap"), ("xmp.tar.gz", images, "*.xmp")):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for p in ([out / "colmap"] if pattern == "colmap" else sorted(root.glob(pattern))):
                tf.add(p, arcname=p.relative_to(root) if pattern == "colmap" else f"xmp/{p.name}")
        (d / name).write_bytes(buf.getvalue())
    (d / "rs_log.txt").write_text(log)
    res = {"tag": tag, "src": src, "images": n, "xmp_priors_in": n_xmp_in, "copy_s": copy_s,
           "seconds": seconds, "code": run["code"], "cmd": cmd, "focal35": focal35,
           "distortion_model": distortion_model, "extra": extra, "post": post,
           "stdout": run["out"][-3000:], "cpu": 8, "memory_mb": 16384}
    (d / "run.json").write_text(json.dumps(res, indent=1))
    vol.commit()
    return {k: v for k, v in res.items() if k != "stdout"}


@app.local_entrypoint()
def run_dir(src: str, tag: str, focal35: float = 0.0, distortion_model: int = 2,
            extra: str = "", post: str = "", timeout: int = 6600, xmp_src: str = "") -> None:
    """`align_dir` on a Volume folder; use with --detach, then
    `modal volume get litereality-tools runs/<tag> <dir>`."""
    import json
    print(json.dumps(align_dir.remote(src, tag, focal35 or None, distortion_model, extra, post,
                                      timeout, xmp_src), indent=1))
