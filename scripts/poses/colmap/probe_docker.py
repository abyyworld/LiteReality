"""Probe the official COLMAP docker image on Modal: CPU CLI options, and whether it starts on a GPU.

    MODAL_IMAGE_BUILDER_VERSION=2025.06 uv run --env-file ~/LiteReality/.env modal run --env main \
        scripts/poses/colmap/probe_docker.py[::gpu_main]
"""

import subprocess

import modal

COLMAP_IMAGE = "colmap/colmap:20261109.9263"
image = modal.Image.from_registry(COLMAP_IMAGE, add_python="3.12")
app = modal.App("poses-colmap-probe-docker", image=image)


@app.function(cpu=2, memory=4096, timeout=600)
def probe():
    cmd = ("colmap -h 2>&1 | head -1; "
           "for c in feature_extractor sequential_matcher global_mapper; do echo == $c; "
           "colmap $c -h 2>&1 | grep -i -E 'type|ImageReader\\.(camera|single)|overlap|loop|vocab|"
           "use_gpu|max_num_features|max_image_size|refine_focal|aliked|lightglue|min_num_matches' ; done; "
           "ls /usr/local/lib | grep -i onnx; ldd $(which colmap) | grep -i -E 'onnx|cudnn|not found'")
    subprocess.run(cmd, shell=True)


@app.function(gpu="L4", cpu=2, memory=4096, timeout=300)
def gpu_probe():
    subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv; env | grep NVIDIA_",
                   shell=True)


@app.function(gpu=["A10G", "T4", "L40S", "A100-40GB"], cpu=2, memory=4096, timeout=300, image=modal.Image.debian_slim())
def gpu_plain():
    subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv", shell=True)


@app.local_entrypoint()
def main():
    probe.remote()


@app.local_entrypoint()
def gpu_main():
    print("plain:", gpu_plain.remote())
    print("colmap image:", gpu_probe.remote())


@app.local_entrypoint()
def plain_main():
    print("plain:", gpu_plain.remote())


@app.function(cpu=1, memory=2048, timeout=300)
def mapper_help():
    subprocess.run("colmap global_mapper -h 2>&1 | grep -- '--'", shell=True)


@app.local_entrypoint()
def mapper_help_main():
    mapper_help.remote()
