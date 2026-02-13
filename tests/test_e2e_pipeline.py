"""E2E smoke tests for PushT and ALOHA sim training pipeline."""

from types import SimpleNamespace
import importlib.util
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_with_lerobot.py"
SPEC = importlib.util.spec_from_file_location("train_with_lerobot", SCRIPT_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class _FakeDataset:
    def __init__(self, dataset_name, delta_timestamps=None, video_backend=None):
        self.dataset_name = dataset_name
        self.num_episodes = 2
        self.num_frames = 20
        self._len = 20

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if "pusht" in self.dataset_name:
            return {
                "observation.image": torch.randn(3, 96, 96),
                "observation.state": torch.randn(2),
                "action": torch.randn(16, 2),
            }
        return {
            "observation.images.cam_high": torch.randn(3, 96, 96),
            "observation.state": torch.randn(14),
            "action": torch.randn(50, 14),
        }


class _FakeMetadata:
    def __init__(self, dataset_name):
        self.fps = 10


def _run_for_dataset(tmp_path, dataset_name):
    MOD.LeRobotDataset = _FakeDataset
    MOD.LeRobotDatasetMetadata = _FakeMetadata

    args = SimpleNamespace(
        dataset=dataset_name,
        profile="quick",
        epochs=1,
        batch_size=2,
        hidden_dim=64,
        max_text_len=16,
        lr=1e-3,
        use_tanh_actions=False,
        output_dir=tmp_path / dataset_name.replace("/", "_"),
    )
    MOD.setup_logging(args.output_dir)
    result = MOD.train(args)
    assert result["best_val_loss"] >= 0
    assert (args.output_dir / "best_model.pt").exists()
    assert (args.output_dir / "results.json").exists()


def test_pusht_pipeline_smoke(tmp_path):
    _run_for_dataset(tmp_path, "lerobot/pusht")


def test_aloha_sim_pipeline_smoke(tmp_path):
    _run_for_dataset(tmp_path, "lerobot/aloha_sim_insertion_human")
