import base64
import hashlib
import importlib.util
import sys
from pathlib import Path


def _module():
    script = Path(__file__).parents[1] / "scripts/download_gcs_prefix.py"
    spec = importlib.util.spec_from_file_location("download_gcs_prefix", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_download_promotes_a_complete_verified_partial(tmp_path: Path) -> None:
    module = _module()
    content = b"complete shard"
    digest = hashlib.md5(content, usedforsecurity=False).digest()
    obj = module.GCSObject(
        name="prefix/shard.tfrecord",
        size=len(content),
        md5=base64.b64encode(digest).decode(),
    )
    partial = tmp_path / ".shard.tfrecord.part"
    partial.write_bytes(content)

    result = module.download_object("unused", obj, "prefix/", tmp_path, retries=0)

    assert result == ("downloaded", len(content))
    assert not partial.exists()
    assert (tmp_path / "shard.tfrecord").read_bytes() == content
