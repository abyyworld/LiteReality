"""Full-body people for the room call: photos of one person -> a skinned 3D Gaussian body.

`lhm` holds the GPU-side inference (LHM++ on Modal, `deploy/modal/avatar/app.py`) and the pure numpy
helpers around it. The room call's own stand-in body and the avatar file format live in
`room_ops/roomcall`, which this package must not import (models sits below room_ops).
"""
