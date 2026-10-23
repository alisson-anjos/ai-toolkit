"""Spatial-only native motion-reference augmentation; no extra RoPE scaling."""
import math
import random


def validate_scales(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError('native reference scales must be a nonempty list')
    result=[]
    for v in values:
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not .125 <= v <= 1:
            raise ValueError('native reference scales must be finite in [0.125, 1]')
        result.append(float(v))
    return tuple(result)


def scale_for_step(scales, seed, step):
    return random.Random(int(seed)+int(step)).choice(validate_scales(scales))


def scaled_native_size(height, width, scale):
    scale=validate_scales([scale])[0]
    if scale == 1:
        return int(height),int(width)
    # Match BFS native_guide_scale's existing floor-to-32 behavior. Actual
    # rounded dimensions and aspect may differ from the nominal scale.
    return max(32,int(height*scale)//32*32),max(32,int(width*scale)//32*32)


def motion_scale(batch, file_item, ref_idx, options):
    if not options.get('enabled',False):
        return 1.0
    source_ids=getattr(file_item,'control_video_source_ids',())
    if ref_idx >= len(source_ids):
        raise ValueError('Multiscale native training requires explicit video source IDs')
    if source_ids[ref_idx] != options['source_channel']:
        return 1.0
    return validate_scales([getattr(batch,'_h3_native_reference_scale',1.0)])[0]
