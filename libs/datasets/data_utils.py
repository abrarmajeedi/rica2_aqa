import os
import copy
import random
import numpy as np
import random
import torch

from fractions import Fraction
from typing import Any, List
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose


def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Encode "EACH VIDEO" one by one and store it. "EACH CLIP" can be accessed according to start time and end time 
class EncodedVideoCached:
    def __init__(self, path, frame_buffer_size=100):
        self.path = path
        self.vid = EncodedVideo.from_path(path, decoder="pyav")
        self.vid._container.seek(0)
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = []
        self.last_t = None

    # this function is used to get "A CLIP" based on start_time and end_time
    def get_clip(self, t1, t2):
        if self.last_t is not None and t1 < self.last_t:
            raise AssertionError("cannot seek backward")
        
        vstream = self.vid._container.streams.video[0]
        vs = vstream.start_time * vstream.time_base
    
        frames = self.get_frames(
            self.vid._container,
            t1 + vs,
            t2 + vs,
            self.frame_buffer,
            self.frame_buffer_size,
        )
        
        self.last_t = t1
        return {
            "num_frames": len(frames),
            # thwc_to_tchw
            "video":
                torch.stack(
                    [torch.from_numpy(frame.to_rgb().to_ndarray()) for frame in frames]
            ).permute(0,3,1,2).to(torch.float32),
            "audio": None,
        }
        
    # this function is used to get "FRAMES" for "A CLIP" based on start_time and end_time
    def get_frames(self, container, t1, t2, buffer, max_buffer_size):
        ret = []
        tb = container.streams.video[0].time_base

        def is_in_range(frame):
            t = frame.pts * tb
            return t >= t1 and t < t2

        def exceeds_range(frame):
            return frame.pts * tb >= t2

        for frame in buffer:
            if is_in_range(frame):
                ret.append(frame)
        
        prev_pts = None
        
        # This try except block is to avoid the EOF error that arrives because t2 exceeds the frame range 
        try:
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise AssertionError("frame is None")
                if prev_pts is not None and frame.pts < prev_pts:
                    raise AssertionError("failed assumption pts in order: ")
                if not isinstance(frame, av.VideoFrame):
                    raise AssertionError("other packets not supported")
                prev_pts = frame.pts

                buffer.append(frame)
                if len(buffer) > max_buffer_size:
                    del buffer[0]

                if is_in_range(frame):
                    ret.append(frame)
                elif exceeds_range(frame):
                    break
        except:
            pass
    
        pts_in_ret = [frame.pts for frame in ret]
        if not (np.diff(pts_in_ret) > 0).all():
            raise AssertionError("not increasing sequence of frames")
        return ret

        
    @property
    def duration(self) -> float:
        vstream = self.vid._container.streams.video[0]
        return vstream.duration * vstream.time_base
