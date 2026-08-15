import importlib.util
from pathlib import Path


def _module():
    script = Path(__file__).parents[1] / "scripts/prepare_droid_statistics.py"
    spec = importlib.util.spec_from_file_location("prepare_droid_statistics", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bounded_parallelism_replaces_autotune_and_caps_explicit_values() -> None:
    module = _module()
    assert module._bounded_parallelism(-1, 16) == 16
    assert module._bounded_parallelism(None, 16) == 16
    assert module._bounded_parallelism(8, 16) == 8
    assert module._bounded_parallelism(32, 16) == 16
