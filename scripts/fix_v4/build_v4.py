"""The atrium retrained with the q_sh3_nogrid preset as a DETACHED Modal job, then cut as usual.

    uv run --env-file .env python scripts/fix_v4/build_v4.py \
        --out run/atrium_stray_q/v4_train --run atrium_v4

This is `litereality splatroom` with the same flags as the fix_v4 retrain (results/fix_v4/retrain/
a1_nogrid.log: stray_atrium_b3, --trainer-preset q_sh3_nogrid, --depth promptda
--depth-loss-dense --normal-lambda 0.05, --train-width 1440), with one change: the trainer. The
CLI's `--ephemeral-trainer` hook is swapped for `DetachedTrainer`, whose `stream(tar, options)`
puts the very tar the pipeline built on a Modal Volume, starts scripts/fix_v4/train_detached.py
with `modal run --detach` (a job that keeps running when this machine's connection drops), and
then only polls the Volume, retrying every read the network fails, until the job writes
done.json. The final snapshot is then handed to the pipeline as if streamed, so the bundle,
masks, physics and report are the pipeline's own.

Run again after a crash of THIS process, it finds the job on the Volume (done.json, or a
progress.json written in the last few minutes) and polls it rather than starting another.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VOLUME_NAME = "litereality-v4-train"
JOB_SCRIPT = REPO / "scripts" / "fix_v4" / "train_detached.py"
# Snapshots the job writes to the Volume (the pipeline is handed only the last; the earlier ones
# are a fallback if the job dies late).
SNAPSHOTS = [2500, 5000, 7500]


def _retry(fn, what: str, tries: int = 40, wait: float = 15.0):
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001 — the network: wait and ask again
            if "not found" in str(exc).lower() or "no such file" in str(exc).lower():
                raise FileNotFoundError(what) from exc
            print(f"    [{what}] attempt {attempt}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == tries:
                raise
            time.sleep(wait)


class DetachedTrainer:
    name = "gsplat-modal (detached job, Volume)"

    def __init__(self, run: str, local_dir: Path, *, max_train_seconds: int = 2100,
                 budget_seconds: int = 2500, poll: float = 20.0) -> None:
        import modal

        self.run, self.local = run, Path(local_dir)
        self.local.mkdir(parents=True, exist_ok=True)
        self.max_train_seconds, self.budget_seconds, self.poll = (max_train_seconds,
                                                                  budget_seconds, poll)
        self.volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True,
                                             environment_name="main")
        self.launcher_log = self.local / "launcher.log"

    # ---- the Volume -------------------------------------------------------------------------
    def _read(self, name: str) -> bytes:
        def go() -> bytes:
            return b"".join(self.volume.read_file(f"{self.run}/{name}"))
        return _retry(go, name)

    def _read_json(self, name: str) -> dict | None:
        try:
            return json.loads(self._read(name))
        except FileNotFoundError:
            return None

    def _upload(self, files: dict[str, Path]) -> None:
        def go() -> None:
            with self.volume.batch_upload(force=True) as batch:
                for remote, local in files.items():
                    batch.put_file(str(local), f"{self.run}/{remote}")
        _retry(go, "upload " + ", ".join(files), tries=20, wait=20.0)

    # ---- the job ----------------------------------------------------------------------------
    def _launch(self) -> subprocess.Popen:
        env = {**os.environ, "MODAL_IMAGE_BUILDER_VERSION": "2025.06", "PYTHONUNBUFFERED": "1"}
        cmd = ["uv", "run", "modal", "run", "--detach", "--env", "main", str(JOB_SCRIPT),
               "--run", self.run, "--max-train-seconds", str(self.max_train_seconds)]
        print(f"    launching: {' '.join(cmd[2:])}", flush=True)
        log = open(self.launcher_log, "a")
        # Its own session: the job's launcher outlives this process (and its terminal).
        return subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def app_id(self) -> str | None:
        try:
            ids = re.findall(r"ap-[A-Za-z0-9]{10,}", self.launcher_log.read_text(errors="replace"))
        except OSError:
            return None
        return ids[-1] if ids else None

    def stop(self) -> None:
        app = self.app_id()
        if app:
            print(f"    stopping {app}", flush=True)
            subprocess.run(["uv", "run", "modal", "app", "stop", app], cwd=REPO, check=False)

    def stream(self, dataset_tar: bytes, options: dict | None = None):
        opts = dict(options or {})
        steps = int(opts.get("steps") or 0)
        opts["snapshots"] = [s for s in SNAPSHOTS if not steps or s < steps]
        t0 = time.perf_counter()
        done = self._read_json("done.json")
        prog = self._read_json("progress.json")
        running = prog is not None and not prog.get("final") and time.time() - prog["time"] < 240
        proc = None
        if done is None and not running:
            tar = self.local / "dataset.tar"
            tar.write_bytes(dataset_tar)
            (self.local / "options.json").write_text(json.dumps(opts, indent=1))
            t = time.perf_counter()
            self._upload({"dataset.tar": tar, "options.json": self.local / "options.json"})
            print(f"    dataset ({len(dataset_tar) / 1e6:.1f} MB) on the Volume in "
                  f"{time.perf_counter() - t:.0f} s", flush=True)
            proc = self._launch()
            launched = time.time()
        else:
            print(f"    the job on the Volume is {'done' if done else 'running'}: polling it",
                  flush=True)
            launched = time.time() - (prog or {}).get("seconds", 0)
        started = False
        last_step = -1
        while done is None:
            time.sleep(self.poll)
            try:
                prog = self._read_json("progress.json")
                done = self._read_json("done.json")
            except Exception as exc:  # noqa: BLE001 — tried 40 times: still only a poll
                print(f"    poll failed: {exc}", flush=True)
                continue
            fresh = prog is not None and prog["time"] >= launched - 5
            if fresh and not started:
                started = True
                yield {"kind": "started", "seconds": prog["seconds"], "gpu": prog.get("gpu"),
                       "images": prog.get("images"), "depths": prog.get("depths")}
            if fresh and prog.get("step", -1) != last_step:
                last_step = prog.get("step", -1)
                left = ((prog.get("steps", 0) - last_step) / prog["it_s"]
                        if prog.get("it_s") else float("nan"))
                print(f"    step {last_step}/{prog.get('steps')} at {prog.get('it_s')} it/s, "
                      f"container {prog['seconds']:.0f} s, ~{left:.0f} s left; snapshots "
                      f"{prog.get('snapshots')}", flush=True)
            if fresh and prog["seconds"] > self.budget_seconds:
                print(f"    over the {self.budget_seconds} s budget: stopping the job", flush=True)
                self.stop()
                break
            if (proc is not None and proc.poll() is not None and not fresh
                    and time.time() - launched > 900):
                yield {"kind": "error", "error": "the detached job never reported progress "
                       f"(launcher exited {proc.returncode}; {self.launcher_log})", "log": ""}
                return
        if done is None:
            done = self._read_json("done.json") or {}
        for name in ("done.json", "progress.json", "trainer.log"):
            try:
                (self.local / name).write_bytes(self._read(name))
            except FileNotFoundError:
                pass
        snaps = sorted(int(s) for s in (done.get("snapshots") or {}))
        if not snaps:
            yield {"kind": "error", "error": f"job ended with {done.get('exit')} and no snapshot",
                   "log": done.get("log_tail", "")}
            return
        step = snaps[-1]
        t = time.perf_counter()
        data = self._read(f"snap_{step}.spz")
        (self.local / f"snap_{step}.spz").write_bytes(data)
        print(f"    snapshot {step} ({len(data) / 1e6:.1f} MB) downloaded in "
              f"{time.perf_counter() - t:.0f} s; waited {time.perf_counter() - t0:.0f} s in all",
              flush=True)
        info = done["snapshots"][str(step)] if str(step) in done["snapshots"] else {}
        yield {"kind": "snapshot", "step": step, "seconds": info.get("seconds", 0.0),
               "splats": info.get("splats"), "spz": data}
        if done.get("exit") != 0 or done.get("missing_snapshots"):
            yield {"kind": "error", "error": f"trainer exited with {done.get('exit')}; missing "
                   f"snapshots {done.get('missing_snapshots')}", "log": done.get("log_tail", "")}
            return
        yield {"kind": "done", "seconds": done.get("total_s", 0.0), "stats": done.get("stats")
               or {}, "timing": {"train_s": done.get("train_s"), "setup_s": done.get("setup_s")}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="run/atrium_stray_q/v4_train")
    ap.add_argument("--scene", default=None, help="default: <out>/physics/scene.xml")
    ap.add_argument("--scan", default=str(Path.home() / "LiteReality-scans" / "stray_atrium_b3"))
    ap.add_argument("--run", default="atrium_v4")
    ap.add_argument("--max-train-seconds", type=int, default=2100)
    ap.add_argument("--budget-seconds", type=int, default=2500)
    args = ap.parse_args()
    os.chdir(REPO)
    out = Path(args.out)
    scene = Path(args.scene) if args.scene else out / "physics" / "scene.xml"
    from litereality_agent import cli

    def detached(stack, modal_kw):        # replaces cli._ephemeral_splat_trainer
        return DetachedTrainer(args.run, out / "trainer", max_train_seconds=args.max_train_seconds,
                               budget_seconds=args.budget_seconds)

    cli._ephemeral_splat_trainer = detached
    return cli.main(["splatroom", args.scan, "--scene", str(scene), "--out", str(out),
                     "--ephemeral-trainer", "--trainer-preset", "q_sh3_nogrid",
                     "--depth", "promptda", "--depth-loss-dense", "--normal-lambda", "0.05",
                     "--train-width", "1440"])


if __name__ == "__main__":
    sys.exit(main())
