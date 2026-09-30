"""Explicit guide roles and shared target/guide spatial transforms."""
import os
from PIL import Image, ImageOps


def image_guide_channel(kwargs):
    value = kwargs.get('image_guide_channel', 1)
    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2, 3):
        raise ValueError('image_guide_channel must be 1, 2, or 3')
    return value


def is_image_guide(path, dataset_config, kwargs):
    if not kwargs.get('align_image_refs', False):
        return False
    channel = image_guide_channel(kwargs)
    named = [getattr(dataset_config, f'control_path_{i}', None) for i in (1, 2, 3)]
    roots = named if any(named) else (getattr(dataset_config, 'control_path', None) or [])
    roots = [roots] if isinstance(roots, str) else roots
    for index, candidate in enumerate(roots, 1):
        if candidate and os.path.dirname(os.path.abspath(path)) == os.path.abspath(candidate):
            role = getattr(dataset_config, f'control_role_{index}', None)
            if role not in (None, 'guide', 'reference'):
                raise ValueError('Control roles must be guide or reference')
            return role == 'guide' if role else index == channel
    return False


def spatial_signature(item):
    if item is None:
        return None
    return tuple(getattr(item, key, 0) for key in (
        'scale_to_width', 'scale_to_height', 'crop_x', 'crop_y',
        'crop_width', 'crop_height', 'flip_x', 'flip_y'))


def prepare_guide_image(image, target_height, target_width, factor=1, item=None):
    """Apply the target's resize/crop/flips, then reduce to its guide canvas."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1:
        raise ValueError('Guide factor must be a positive integer')
    if target_height % (32 * factor) or target_width % (32 * factor):
        raise ValueError('Aligned guide target dimensions must be multiples of 32 * factor')
    image = ImageOps.exif_transpose(image).convert('RGB')
    if item is not None:
        if getattr(item, 'flip_x', False):
            image = ImageOps.mirror(image)
        if getattr(item, 'flip_y', False):
            image = ImageOps.flip(image)
        sw, sh, x, y, cw, ch, _, _ = spatial_signature(item)
        if not sw or not sh or (cw, ch) != (target_width, target_height):
            raise ValueError('Aligned guide requires the target bucket resize/crop geometry')
        image = image.resize((sw, sh), Image.Resampling.BICUBIC)
        image = image.crop((x, y, x + cw, y + ch))
    else:
        # Sampling has no dataset crop: explicitly fit the source to the output canvas.
        image = ImageOps.fit(image, (target_width, target_height), Image.Resampling.BICUBIC)
    return image.resize((target_width // factor, target_height // factor), Image.Resampling.LANCZOS)
