"""Instruction-driven image editing (Qwen-Image-Edit-2509) for reference-image generation.

``inference`` runs inside the Modal container and keeps every heavy import lazy; ``modal`` is the
client that the reference step calls when ``LR_IMAGE_PROVIDER=modal``.
"""
