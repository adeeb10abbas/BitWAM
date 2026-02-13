# 1bit_vla

`1bit_vla` is a research codebase for **1-bit-ready Vision-Language-Action (VLA)** models.

This repository now exposes one canonical architecture and one canonical training pipeline:

- **Model**: `VLABitNet` (`vision + language + state -> action`)
- **Tasks**: `lerobot/pusht` and `lerobot/aloha_sim_insertion_human`
- **Pipeline**: `scripts/train_with_lerobot.py` (pip-first, no local `../lerobot` clone required)

## Architecture

Core modules:

- `src/bit_vla/models/vision_encoder.py`: CNN + BitLinear projections
- `src/bit_vla/models/language_encoder.py`: token embedding + BitLinear self-attention block
- `src/bit_vla/models/action_decoder.py`: BitLinear MLP + FP action head
- `src/bit_vla/models/vla_bitnet.py`: multimodal fusion and canonical forward contract

Canonical forward contract for `VLABitNet`:

- `images`: `[B, C, H, W]` or `[B, T, C, H, W]`
- `token_ids`: `[B, L]`
- `attention_mask` (optional): `[B, L]` bool
- `states` (optional when `state_dim > 0`): `[B, D]` or `[B, T, D]`
- output: `actions` `[B, action_dim]`

## Install

```bash
pip install -e .
pip install lerobot
```

## End-to-End Training

Quick smoke run:

```bash
python scripts/train_with_lerobot.py --config configs/pusht_quick.yaml
python scripts/train_with_lerobot.py --config configs/aloha_sim_quick.yaml
```

Longer run:

```bash
python scripts/train_with_lerobot.py --config configs/pusht_standard.yaml
python scripts/train_with_lerobot.py --config configs/aloha_sim_standard.yaml
```

Outputs are written under `outputs/vla_pipeline/<dataset>_<profile>/`:

- `train.log`
- `best_model.pt`
- `results.json`

## One-command helper

```bash
./run_lerobot_examples.sh setup
./run_lerobot_examples.sh gpu_info
./run_lerobot_examples.sh smoke_pusht
./run_lerobot_examples.sh smoke_aloha
./run_lerobot_examples.sh smoke_pusht_gpu
./run_lerobot_examples.sh smoke_aloha_gpu
./run_lerobot_examples.sh rollout_pusht
./run_lerobot_examples.sh rollout_aloha
```

## Rollout-style verification

After training, run rollout-style evaluation:

```bash
python scripts/rollout_eval.py \
  --config configs/pusht_quick.yaml \
  --checkpoint outputs/vla_pipeline/lerobot_pusht_quick/best_model.pt
```

This produces `rollout_report.json` containing:

- `mean_mse`
- `mean_l1`
- `within_threshold_percentage`
- `status` (`PASS` or `WARN`)

## Tests

```bash
pytest tests/test_models.py tests/test_training.py tests/test_utils.py tests/test_e2e_pipeline.py
```
