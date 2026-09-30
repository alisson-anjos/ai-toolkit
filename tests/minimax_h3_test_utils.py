"""Load actual CPU geometry/helpers without importing every training model.

Model methods are compiled from their source AST unchanged. This isolates unit
coverage from heavyweight model-loading dependencies; it is not a training test.
"""
import ast
import importlib.util
import sys
from PIL import Image
from toolkit.h3_reference_rope import options_from_kwargs, PHASE_VERSION

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
helper_spec = importlib.util.spec_from_file_location('_h3_aligned_guides', ROOT / 'toolkit/aligned_guides.py')
aligned_guides = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(aligned_guides)
prepare_guide_image = aligned_guides.prepare_guide_image
spatial_signature = aligned_guides.spatial_signature
image_guide_channel = aligned_guides.image_guide_channel
is_image_guide = aligned_guides.is_image_guide
MODEL = ROOT / 'extensions_built_in/diffusion_models/minimax_h3/minimax_h3.py'
SRC = MODEL.parent / 'src'

spec = importlib.util.spec_from_file_location('_minimax_h3_test_packing', SRC / 'packing.py')
packing = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = packing
spec.loader.exec_module(packing)


def source_function(path, name, namespace, class_name=None):
    tree = ast.parse(Path(path).read_text())
    body = tree.body
    if class_name:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def model_method(name, class_name='MinimaxH3Model', **namespace):
    return source_function(MODEL, name, {'options_from_kwargs': options_from_kwargs, 'PHASE_VERSION': PHASE_VERSION, 'packing': packing, 'Image': Image, 'prepare_guide_image': prepare_guide_image,
                                          'spatial_signature': spatial_signature, 'image_guide_channel': image_guide_channel,
                                          'is_image_guide': is_image_guide, **namespace}, class_name)
