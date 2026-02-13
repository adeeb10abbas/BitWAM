# LeRobot Integration (Pip-First)

This integration path is now **pip-first** and supports two canonical simulation tasks:

- `lerobot/pusht`
- `lerobot/aloha_sim_insertion_human`

No local `../lerobot` clone is required.

## Setup

```bash
pip install -e .
pip install lerobot
```

## Quick validation runs

```bash
./run_lerobot_examples.sh gpu_info
python scripts/train_with_lerobot.py --config configs/pusht_quick.yaml
python scripts/train_with_lerobot.py --config configs/aloha_sim_quick.yaml
```

Force NVIDIA GPU:

```bash
python scripts/train_with_lerobot.py --config configs/pusht_quick.yaml --device cuda
```

## Standard training runs

```bash
python scripts/train_with_lerobot.py --config configs/pusht_standard.yaml
python scripts/train_with_lerobot.py --config configs/aloha_sim_standard.yaml
```

## Rollout-style checkpoint verification

```bash
python scripts/rollout_eval.py \
  --config configs/pusht_quick.yaml \
  --checkpoint outputs/vla_pipeline/lerobot_pusht_quick/best_model.pt
```

The script writes `rollout_report.json` and prints a summary with:

- `mean_mse` and `mean_l1` over sampled batches
- `within_threshold_percentage`
- `status` (`PASS` when `mean_l1 <= pass_l1_threshold`, else `WARN`)

## Real language conditioning

The pipeline uses deterministic task prompts and tokenization for sim training:

- PushT prompt: pushing and stable motion objective
- ALOHA sim prompt: bimanual insertion objective

Tokenization is implemented in `src/bit_vla/utils/text.py`.

## Output artifacts

Each run creates:

- `train.log`
- `best_model.pt`
- `results.json`

under:

- `outputs/vla_pipeline/lerobot_pusht_<profile>/`
- `outputs/vla_pipeline/lerobot_aloha_sim_insertion_human_<profile>/`

## Troubleshooting

- If `ImportError: lerobot`:
  - run `pip install lerobot`
- If dataset/video backend errors:
  - ensure `av` is installed and Python environment matches your `pip install -e .`
- If out-of-memory:
  - lower `--batch_size` and/or `--hidden_dim`
