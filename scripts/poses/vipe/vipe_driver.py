"""ViPE entry point used inside the Modal container (copied next to ViPE's own run.py).

Same as ViPE's run.py, plus one optional patch: when VIPE_FIXED_FX is set (fx in pixels at the
1080 x 1920 benchmark size), GeoCalib is replaced by a fixed pinhole focal, and the caller passes
pipeline.init.intrinsics=gt so SLAM does not optimise the intrinsics.
"""

import math
import os

import hydra
from omegaconf import DictConfig

FULL_HEIGHT = 1920

if os.environ.get("VIPE_FIXED_FX"):
    import vipe.pipeline.default as vipe_default
    from vipe.pipeline.processors import IntrinsicEstimationProcessor
    from vipe.utils.cameras import CameraType

    class FixedIntrinsicsProcessor(IntrinsicEstimationProcessor):
        def __init__(self, video_stream, gap_sec=1.0, camera_type=CameraType.PINHOLE, model_cache=None):
            super().__init__(video_stream, gap_sec)
            h, _ = video_stream.frame_size()
            fx = float(os.environ["VIPE_FIXED_FX"]) * h / FULL_HEIGHT
            self.fov_y = 2.0 * math.atan(h / (2.0 * fx))
            self.camera_type = camera_type
            print(f"[driver] fixed intrinsics: fx={fx:.2f} px at height {h}")

    vipe_default.GeoCalibIntrinsicsProcessor = FixedIntrinsicsProcessor


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(args: DictConfig) -> None:
    from vipe.config import validate_typed_config
    from vipe.streams.base import StreamList

    typed_args = validate_typed_config(args)
    stream_list = StreamList.make(typed_args.streams)

    from vipe.pipeline import make_pipeline
    from vipe.utils.logging import configure_logging

    pipeline = make_pipeline(typed_args.pipeline)
    logger = configure_logging()
    for stream_idx in range(len(stream_list)):
        video_stream = stream_list[stream_idx]
        logger.info(f"Processing {video_stream.name()} ({stream_idx + 1} / {len(stream_list)})")
        pipeline.run(video_stream)
        logger.info(f"Finished processing {video_stream.name()}")


if __name__ == "__main__":
    run()
