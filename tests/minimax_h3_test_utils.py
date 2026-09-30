"""Load actual CPU geometry/helpers without importing every training model.

Model methods are compiled from their source AST unchanged. This isolates unit
coverage from heavyweight model-loading dependencies; it is not a training test.
"""
import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
    return source_function(MODEL, name, {'packing': packing, **namespace}, class_name)
