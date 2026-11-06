import hashlib
import importlib.util
from pathlib import Path

import pytest


def load_demo():
    spec = importlib.util.spec_from_file_location("gbs_demo", Path(__file__).parents[1]/"examples/belief_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_download_cache_and_checksum(tmp_path, monkeypatch):
    demo = load_demo()
    contents = b"synthetic checkpoint contents"
    monkeypatch.setattr(demo, "DEMO_SHA256", hashlib.sha256(contents).hexdigest())
    downloads = []

    def download(url, filename, *, hash_prefix, progress):
        downloads.append(url)
        assert hash_prefix == demo.DEMO_SHA256
        Path(filename).write_bytes(contents)

    monkeypatch.setattr(demo.torch.hub, "download_url_to_file", download)
    checkpoint = demo.demo_checkpoint(tmp_path/"checkpoints")
    assert checkpoint.read_bytes() == contents
    assert demo.demo_checkpoint(tmp_path/"checkpoints") == checkpoint
    assert len(downloads) == 1
    checkpoint.write_bytes(b"modified file")
    with pytest.raises(ValueError, match="checksum mismatch"):
        demo.demo_checkpoint(tmp_path/"checkpoints")
