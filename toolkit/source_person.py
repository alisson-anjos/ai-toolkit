"""Source-person photographs are encoder context, never replacement controls."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
from PIL import Image


@dataclass
class SourcePersonRef:
    image: Image.Image


def source_person_photo(folder, target):
    if not folder:
        return None
    stem = Path(target).stem
    matches = [p for p in Path(folder).glob(stem + '.*')
               if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp')]
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one source-person photo for {stem}, found {len(matches)}')
    return str(matches[0])


def source_person_cache_info(folder, target):
    path = source_person_photo(folder, target)
    if path is None:
        return {}
    return {'source_person_context_version': 1,
            'source_person_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
