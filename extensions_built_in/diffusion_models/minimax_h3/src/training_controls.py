"""Reference dropout that shares one selection across all passes of a batch."""
import math
import torch


def dropout_probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a probability between 0 and 1')
    return float(value)


def reference_keep_masks(batch, image_count, video_count, image_probability, video_probability, image_guide_flags=(), video_guide_flags=()):
    signature = (image_count, video_count, image_probability, video_probability, tuple(image_guide_flags), tuple(video_guide_flags))
    cached = getattr(batch, '_h3_reference_keep', None)
    if cached is not None:
        if cached[0] != signature:
            raise ValueError('Reference counts changed between passes of the same training batch')
        return cached[1], cached[2]
    def draw(count, probability):
        if probability == 0:
            return (True,) * count
        if probability == 1:
            return (False,) * count
        return tuple(bool(v) for v in torch.rand(count) >= probability)
    images = (tuple(draw(1, video_probability if guide else image_probability)[0]
                    for guide in image_guide_flags) if image_guide_flags else draw(image_count, image_probability))
    videos = (tuple(draw(1, video_probability if guide else image_probability)[0]
                    for guide in video_guide_flags) if video_guide_flags else draw(video_count, video_probability))
    batch._h3_reference_keep = (signature, images, videos)
    return images, videos

